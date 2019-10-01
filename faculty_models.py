# Copyright 2019 Faculty Science Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import posixpath

from six.moves import urllib
import faculty
import mlflow.pyfunc
import mlflow.tracking.artifact_utils


def download(project_id, model_id, path=None, version_number=None):
    artifact_path = _determine_artifact_path(
        project_id, model_id, path, version_number
    )
    return mlflow.tracking.artifact_utils._download_artifact_from_uri(
        artifact_path
    )


def load_mlmodel(project_id, model_id, path=None, version_number=None):
    artifact_path = _determine_artifact_path(
        project_id, model_id, path, version_number
    )
    return mlflow.pyfunc.load_model(artifact_path)


def _determine_artifact_path(
    project_id, model_id, path=None, version_number=None
):
    model_version = _get_model_version(project_id, model_id, version_number)
    if path is None:
        return model_version.artifact_path
    else:
        return _append_subpath_to_uri(model_version.artifact_path, path)


def _get_model_version(project_id, model_id, version_number=None):
    client = faculty.client("model")
    versions = client.list_versions(project_id, model_id)
    if version_number is None:
        matching = versions[-1:]
    else:
        matching = [v for v in versions if v.version_number == version_number]

    try:
        [version] = matching
        return version
    except ValueError:
        if len(matching) == 0:
            tpl = "No version of model {} with version number {} found"
        else:
            tpl = "Multiple versions of model {} with version number {} found"
        raise ValueError(tpl.format(model_id, version_number))


def _append_subpath_to_uri(uri, subpath):
    parsed_uri = urllib.parse.urlparse(uri)
    modified_uri = parsed_uri._replace(
        path=posixpath.join(parsed_uri.path, subpath)
    )
    return urllib.parse.urlunparse(modified_uri)
