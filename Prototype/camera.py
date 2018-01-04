import picamera

def takePhoto(filename):
    camera = picamera.PiCamera()
    camera.start_preview()
    camera.capture(filename)
