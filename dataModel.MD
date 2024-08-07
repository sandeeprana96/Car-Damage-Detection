# Car Damage Estimation

## Damage Found In Car

```json
{
  "count": 2,
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
    },
    {
      "damage_type": "scratch",
      "confidence": 88.87399911880493,
      "box_points": [
        343,
        205,
        400,
        228
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

## No Damage Found in car

```json
{
  "count": 0,
  "damage_info": [],
  "input_image": "bumperWithNoDamage.png",
  "output_image": "detection-bumperWithNoDamage.png",
  "input_path": "/input/",
  "output_path": "/Car_Damage_Detection/output/Detection-bumperWithNoDamage.png",
  "success": true
}
```

## Error Scenario: File not found
```json
{
  "success": false,
  "error_message": "Please check if image file path or/and image name are correct"
}
```

## Other Errors
```json
{
  "success": false,
  "error_message": Custom Error Message
}
```
