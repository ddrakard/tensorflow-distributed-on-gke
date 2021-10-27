import os
from pathlib import Path

import typing

import keras
from google.cloud import storage


class ModelUploader:
    def __init__(
            self, bucket_name: str, bucket_base_folder_name: str,
            google_cloud_access_key_path: str, local_temporary_directory: str):
        """
            A class to help uploading snapshots to google cloud storage during
            training.

            :param bucket_name: Name of the google cloud bucket to upload to.
            :param bucket_base_folder_name: Name of the folder to upload to.
                If it exists the name will be changed.
            :param google_cloud_access_key_path: Path to credentials JSON file
                for google cloud.
            :param local_temporary_directory: A directory to generate model
                snapshots in.
        """
        self.bucket_name = bucket_name
        self.key_path = google_cloud_access_key_path
        self.local_directory = new_directory_name(
            local_temporary_directory + '/model_uploader')
        os.makedirs(self.local_directory)
        self.cloud_base_folder_name = bucket_base_folder_name
        self.cloud_root_folder = None
        self.last_upload_folder = None

    def take_snapshot(self, model: keras.Model) -> None:
        """
            Save the model. The first time the whole model will be saved, after
            this only the weights will be saved.

            Note when loading weights, "/model_weights" must follow the
            directory where they are saved.
        """
        if self.cloud_root_folder is None:
            self._initialize_in_cloud(model)
        else:
            # TODO: This is not efficient as all existing snapshots are
            #  queried individually
            folder = upload_weights(
                model, self.bucket_name,
                self.cloud_root_folder + '/weights_snapshot', self.key_path,
                self.local_directory + '/weights_snapshots')
            self.last_upload_folder = folder

    def last_upload_location(self) -> str:
        """
            Description of the last snapshot upload location.
        """
        return (
            'google cloud storage /' + self.bucket_name + '/'
            + self.last_upload_folder)

    def _initialize_in_cloud(self, model: keras.Model) -> None:
        """
            Prepare a suitable folder in google cloud storage and do the first
            upload.
        """
        model.save(
            Path(self.local_directory + '/initial_model').absolute())
        self.cloud_root_folder = safe_upload_directory(
            self.local_directory, self.bucket_name,
            self.cloud_base_folder_name, self.key_path)
        self.last_upload_folder = self.cloud_root_folder + '/initial_model'


def upload_weights(
        model, bucket_name: str, bucket_folder: str, access_key_path: str,
        local_temporary_directory: str):
    """
        Note when loading weights, /model_weights must follow the directory
        where they are saved.
    """
    save_directory = new_directory_name(
        local_temporary_directory + '/model_weights')
    # The additional "/model_weights" is the prefix Tensorflow uses for
    #  filenames.
    model.save_weights(save_directory + '/model_weights')
    return safe_upload_directory(
        save_directory, bucket_name, bucket_folder, access_key_path)


def storage_client(access_key_path):
    if not hasattr(storage_client, 'result'):
        storage_client.result = (
            storage.Client.from_service_account_json(access_key_path))
    return storage_client.result


def safe_upload_directory(
        local_directory: str, bucket_name: str, bucket_folder: str,
         access_key_path: str):
    new_folder_name = new_cloud_folder_name(
        bucket_name, bucket_folder, access_key_path)
    upload_directory_to_cloud_storage(
        local_directory, bucket_name, new_folder_name, access_key_path)
    return new_folder_name


def upload_directory_to_cloud_storage(
         local_directory: str, bucket_name: str, bucket_folder: str,
         access_key_path: str):
    bucket = storage_client(access_key_path).get_bucket(bucket_name)
    root = Path(local_directory)
    for child in root.rglob("*"):
        if child.is_file():
            relative_path = str(child.relative_to(root))
            blob = bucket.blob(bucket_folder + '/' + relative_path)
            blob.upload_from_filename(str(child.absolute()))


def new_cloud_folder_name(
        bucket_name: str, folder_base_name: str, access_key_path: str)    :

    def check_folder_exists(name):
        contents = (
            storage_client(access_key_path)
            .list_blobs(bucket_name, max_results=1, prefix=name))
        for _ in contents:
            return True
        return False

    return unique_name(folder_base_name, check_folder_exists)


def new_directory_name(base_name: str):
    return unique_name(base_name, lambda name: Path(name).exists())


# TODO: Convert to a class so that after it has been initialized, new names
#   can be generated without doing potentially slow lookups
def unique_name(
        prefix: str, check_name_exists: typing.Callable[[str], bool],
        suffix: str = '', index_separator: str = '_',
        number_first_name: bool = False) -> str:
    """
        Create a unique name from a base name. This assists in avoiding
        name collisions during creation of assets.

        :param prefix: The basic desired name. If it already exists a unique
            postfix will be added.
        :param check_name: A callback to check whether a generated name exists.
        :param suffix: Any constant part at the end of the name, which the
            index should be placed in front of.
        :param index_separator: A character to place between the prefix and
            any appended number, if a number is appended.
        :param number_first_name: No similar name exists, if True the name will
            be numbered "1", if False it will have no number.
        :return: An previously unused name.
    """
    # TODO: The bare name should not be used if it or _0 or _1 are present,
    # and _1 should not be used if the bare name or _1 are present.
    index = 1
    while True:
        if index == 1 and not number_first_name:
            name = prefix + suffix
        else:
            name = prefix + index_separator + str(index) + suffix
        if check_name_exists(name):
            index += 1
        else:
            return name
