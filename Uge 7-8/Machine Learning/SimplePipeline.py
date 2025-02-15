import depthai as dAi
import cv2 





# Opretter pipeline
pipeline = dAi.Pipeline()

# Opret kameraenhed (Mono kamera + RGB kamera)
cam_rgb = pipeline.create(dAi.node.ColorCamera)
cam_rgb.setBoardSocket(dAi.CameraBoardSocket.RGB)
cam_rgb.setResolution(dAi.ColorCameraProperties.SensorResolution.THE_1080P)
cam_rgb.setFps(30)


# Opret en XLinkOut node - XLinkOut bruges til at sende data (i dette tilfælde billeder) ud af systemet
xout = pipeline.create(dAi.node.XLinkOut)
xout.setStreamName("video")  # Giv outputstrømmen et navn (her kaldet "video")

# Link kameraets output til XLinkOut input - Dette betyder, at billederne fra kameraet sendes til XLinkOut
cam_rgb.out.link(xout.input)

# Start DepthAI-enheden med den pipeline, vi har oprettet
with dAi.Device(pipeline) as device:
    q = device.getOutputQueue(name="video", maxSize=8, blocking=False)

    while True:
        
         # Hent det næste billede fra output køen. `frame` er en DepthAI frame objekt.
        frame = q.get()

        # Få billedet som et OpenCV-billede (så vi kan vise det med OpenCV) - Konverter til openCV
        frame_data = frame.getCvFrame()

        
        cv2.imshow("RGB", frame_data) # Viser billedet i et vindue


        # Break loop = q
        if cv2.waitKey(1) == ord('q'):
            break 

    # Luk OpenCV vinduet når programmet afsluttes
    cv2.destroyAllWindows()