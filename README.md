# Car Damage Detection
This car damage detection model detects the external damage on the car in the form of scratch or dent. The model is trained with 815 images of various damaged car parts and used transfer learning on top of the YOLO-v3 model for training.
 
Damaged Car Image (Input to the model):

![Damaged Car](https://i.ibb.co/DCSLb1b/Image7.png=250x)

Damaged Car Image (Input to the model):

![Damage Detected using Car Damage Detection Model](https://i.ibb.co/Ld9f81z/Detection-Image7.png=250x)


## Installation

Install Dependencies by running:

```bash
pip install -r requirements.txt
```

## Usage

Refer to `usage.py` for sample usage.
The below method inside `/detection` directory, takes 3 parameters `imagePath`, `imageName` & `threshold` as:
```python
detect(imagePath, imageName, threshold = 0.3)
```
`imagePath`: Path of the image of the damaged car.

`imageName`: Name of the Image.

`threshold`: Default: 0.3 (Threshold of the probability to include detection or retry with a different image.

`OUTPUT` Folder: Holds the image with a bounding box around the input image.

`RESPONSE`: refer to `dataModel.MD` file for detailed response. Sample response shown below:
```json
{
  "count": 1,
  "damage_info": [
    {
      "damage_type": "scratch",
      "confidence": 92.35874056816101,
      "box_points": [
        326,
        213,
        539,
        252
      ]
    }
  ],
  "input_image": "left_door.jpg",
  "output_image": "detection-left_door.jpg",
  "input_path": "/input/",
  "output_path": "/Car_Damage_Detection/output/detection-left_door.jpg",
  "success": true
} 
```

## Contributors
Dhanush Kumar (dhanush.kumar@globallogic.com)

Vashu Raghav (vashu.raghav@globallogic.com)

Muhammad Hamzah (muhammmad.hamzah@globallogic.com)

