import ceeri_lop
import cv2

obama_image = ceeri_lop.loadImageFile("Obama.jpg")
obama_face_encoding = ceeri_lop.f_ecs(obama_image)[0]
harvey_image = ceeri_lop.loadImageFile("Harvey.jpg")
harvey_face_encoding = ceeri_lop.f_ecs(harvey_image)[0]

video = cv2.VideoCapture(0) #1

positionsOfFace = []   # locations in a face
profile_enc = []      # enocdings in a face
matchedNames = []       # To display names when a face is matched
alternate = True   # To process every alternate frame in the video to make recognition faster


tomatchfaces = [harvey_face_encoding,obama_face_encoding]  #faces to be matched

while True:
    ret, frame = video.read()

    smallFrame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if alternate:
        positionsOfFace = ceeri_lop.facepos(smallFrame)
        profile_enc = ceeri_lop.f_ecs(smallFrame, positionsOfFace)

        matchedNames = []
        for face_encoding in profile_enc:
            match = ceeri_lop.disgface(tomatchfaces, face_encoding)
            found = "Unknown"

            if match[0]:
                found = "harvey"
            elif match[1]:
                found = "obama"

            matchedNames.append(found)

    alternate = not alternate

    for (top, right, bottom, left), name in zip(positionsOfFace, matchedNames):   # to make the rectangle around the matched faces
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
