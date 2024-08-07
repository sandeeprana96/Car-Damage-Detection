from imageai.Detection.Custom import CustomObjectDetection
import tensorflow as tf
import os
import json
import cv2

import cv2

def rescale_image(input_path,input_image_name):
    img = cv2.imread(input_path+input_image_name, 1)
    if(img.shape[1] > 640 and img.shape[0] > 480):
        resized_image = cv2.resize(img, (640, 480))
    else:
        resized_image = img
    newImagePath = os.path.join(os.getcwd(), "raw_images/"+input_image_name)
    cv2.imwrite(newImagePath, resized_image)



## takes path of the image & name of the image (with extension) and saved it to the Output folder with name Detect-ImageName
def detect(imagePath, imageName, threshold = 30):
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        execution_path = os.getcwd()
        tf.compat.v1.Session()
        if imagePath.endswith('/') is not None:
            imagePath = imagePath+"/"
        rescale_image(imagePath,imageName)
        newImagePath = os.path.join(execution_path, "raw_images/"+imageName)
        detector = CustomObjectDetection()
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(
            execution_path, "model/models/detection_model-ex-005--loss-0013.003.h5"))
        detector.setJsonPath(os.path.join(
            execution_path, "model/json/detection_config.json"))
        detector.loadModel()
        outputImageName = "detection-" +  imageName;
        output_image_path=os.path.join(execution_path, "output/"+ outputImageName)
        detections = detector.detectObjectsFromImage(input_image=newImagePath, output_image_path=os.path.join(execution_path, "output/"+outputImageName), minimum_percentage_probability=threshold)
        count = len(detections)
        response = {'count': count}

        img = cv2.imread(newImagePath, 1)
        damage_info = []
        for damage in detections:
            damage_info.append(
                {'damage_type': damage['name'], 'confidence': damage['percentage_probability'], 'box_points': damage['box_points']})

            x1, y1, x2, y2 = damage['box_points']
            damage_type = damage['name']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, damage_type, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        response['damage_info'] = damage_info
        response['input_image'] = imageName
        response['output_image'] = outputImageName
        response['input_path'] = imagePath
        response['output_path'] = os.path.join(execution_path, "output/"+outputImageName)
        response['success'] = True
        cv2.imwrite(response['output_path'], img)
        return json.dumps(response)
    except AttributeError:
        errorResponse = {}
        errorResponse['success'] = False
        errorResponse['error_message'] = "Please check if image file path or/and image name are correct"
        print("Error: Please check if image file path or/and image name are correct");
        return json.dumps(errorResponse)
    except Exception as e:
        errorResponse = {}
        print("Error : ", str(e))
        errorResponse['success'] = False
        errorResponse['error_message'] = e
        return json.dumps(errorResponse)

