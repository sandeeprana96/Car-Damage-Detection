from Detection.detect import detect

## takes path of the image & name of the image (with extension) and saved it to the Output folder with name Detect-ImageName
def detectCarAccidentImage(imagePath, imageName):
    response = detect(imagePath, imageName)
    print(response)
    return response


detectCarAccidentImage("/Users/muhammad.hamzah/Downloads", "cs1.jpg")
