import argparse
import base64
import json
import re
import requests
# [START import_libraries]
import googleapiclient.discovery
# [END import_libraries]


# [START predict_json]
def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']
# [END predict_json]


def main(project, model, filename, version=None, force_tfrecord=False):
    """Send user input to the prediction service."""
    # check if filename is url or image_file
    try:
        z = re.match("^(http|https)://",filename )
        if z:
            img = base64.b64encode(requests.get(filename).content).decode("utf-8")
        else:
            img = base64.b64encode(open(filename, "rb").read()).decode("utf-8")
        jn = json.dumps({"key": "0", "image_bytes": {"b64": img}})
        user_input = json.loads(jn)
    except KeyboardInterrupt:
        return

    if not isinstance(user_input, list):
        user_input = [user_input]
    try:
        result = predict_json(
            project, model, user_input, version=version)
    except RuntimeError as err:
        print(str(err))
    else:
        print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--project',
        help='Project in which the model is deployed',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model',
        help='Model name',
        type=str,
        required=True
    )
    parser.add_argument(
        '--filename',
        help='Name of the image.',
        type=str,
        required=True
    )
    parser.add_argument(
        '--version',
        help='Name of the version.',
        type=str
    )
    args = parser.parse_args()
    main(
        args.project,
        args.model,
        args.filename,
        version=args.version
    )
