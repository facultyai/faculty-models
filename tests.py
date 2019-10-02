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


import uuid

import faculty_models


PROJECT_ID = uuid.uuid4()
MODEL_ID = uuid.uuid4()


def test_download(mocker):
    model_versions = [mocker.Mock(version_number=i) for i in range(5)]

    mock_client = mocker.Mock()
    mock_client.list_versions.return_value = model_versions
    mocker.patch("faculty.client", return_value=mock_client)

    mlflow_download_mock = mocker.patch(
        "mlflow.tracking.artifact_utils._download_artifact_from_uri"
    )

    returned_path = faculty_models.download(PROJECT_ID, MODEL_ID)

    assert returned_path == mlflow_download_mock.return_value

    mock_client.list_versions.assert_called_once_with(PROJECT_ID, MODEL_ID)
    mlflow_download_mock.assert_called_once_with(
        model_versions[-1].artifact_path
    )
