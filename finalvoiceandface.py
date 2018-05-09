import cv2
import pyttsx3
import copy
from pyfcm import FCMNotification
import os
import nexmo
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText



subjects = ["","saurabh verma", "divij jain"]


def findtext():
    from PIL import Image
    import pytesseract
    import cv2
    import os
    import pyttsx3
    t = input()
    
    
    filepath="C:/Users/User/Desktop/idcard/test"+t+".jpg"
 
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 

    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
 

 
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)
    text = pytesseract.image_to_string(Image.open(filename))
    os.remove(filename)
    list=[]
    x=text.find("En")-2
    

    if x>0:
        while text[x] !="\n" and x >=0:
            
            list.append(text[x])
            x-=1
        print(list)
        list.reverse()
        s = "unknown person is identified as "+"".join(list)
        u="".join(list)

        engine = pyttsx3.init()
        engine.setProperty('rate',125)
        engine.say(s)
        engine.runAndWait()
        return(u)
    else:
        print("no name found") 



def newest(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)




def detect_face(img):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    face_cascade = cv2.CascadeClassifier('C:/Users/User/Desktop/face recognition/opencv-files/lbpcascade_frontalface.xml')

    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    
    if (len(faces) == 0):
        print("No face detected")
        return None,None
    
   
    (x, y, w, h) = faces[0]
    
   
    return gray[y:y+w, x:x+h], faces[0]



def prepare_training_data(data_folder_path):
    
    
    dirs = os.listdir(data_folder_path)
    
    
    faces = []
    
    labels = []
    
   
    for dir_name in dirs:
        
        
        if not dir_name.startswith("s"):
            continue;
            
       
        label = int(dir_name.replace("s", ""))
        
        
        subject_dir_path = data_folder_path + "/" + dir_name
        
        
        subject_images_names = os.listdir(subject_dir_path)
        
        
        for image_name in subject_images_names:
            
           
            image_path = subject_dir_path + "/" + image_name

          
            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
            cv2.waitKey(1)
            
            
            face, rect = detect_face(image)
            
            
            if face is not None:
                
                faces.append(face)
                
                labels.append(label)
            
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    
    return faces, labels



print("Preparing data...")
faces, labels = prepare_training_data("C:/Users/User/Desktop/face recognition/training-data")
print("Data prepared")


face_recognizer = cv2.face.LBPHFaceRecognizer_create()


face_recognizer.train(faces, np.array(labels))



def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

def predict(test_img):
    
    img = copy.copy(test_img)
    
    face, rect = detect_face(img)
    
    


    try:
        label, confidence = face_recognizer.predict(face)
        label_text = subjects[label]
    except:
        
        return img
    
    print(confidence)    
    if confidence>29:
        label_text ="unknown"
        draw_rectangle(img, rect)

        draw_text(img, label_text, rect[0], rect[1]-5)
        engine = pyttsx3.init()
        engine.setProperty('rate',125)
        engine.say('Please provide a  piture of your id card')
        engine.runAndWait()
        id=findtext()
        print(id)
    
    
    
    
    
    if label_text=="unknown":
        mailtex="unknown person is at the door and is identified as " + id
        print(mailtex)
    else:
        mailtex=label_text + " is at the door"
    push_service = FCMNotification(api_key=" AAAAemTQsy8:APA91bEdTeWSXCM917MB05lYnIyDN-_7S724mZx4ajEM7mZ3QRBTShpJYdnyk_XcfMR7FamHqtfXRbUvK2WrB5MCeYSBthxMRzH528ouCbkHSXe4cjZQXJzyXiu9hlOuqRMBmfBcn4j8")



    push_service = FCMNotification(api_key=" AAAAemTQsy8:APA91bEdTeWSXCM917MB05lYnIyDN-_7S724mZx4ajEM7mZ3QRBTShpJYdnyk_XcfMR7FamHqtfXRbUvK2WrB5MCeYSBthxMRzH528ouCbkHSXe4cjZQXJzyXiu9hlOuqRMBmfBcn4j8")

    registration_id = "cHjuErYjznw:APA91bERE1tRG5GrWb2kGA2pbTa6cxYfNdTx0WjH-Dc1rGHt8GlI-tQ1aF8Ll9s8GHrQboIB2ZMCAhxnCBKY9YD4iV1Au92GJoIGjFalauVhi6s7GnLOQchYrdMgg_gJMQI5Mx1vD_GP"
    message_title = "Door update"
    result = push_service.notify_single_device(registration_id=registration_id, message_title=message_title, message_body=mailtex)
    
    
    
    '''
    client = nexmo.Client(key='6edceed9', secret='aBYTY7aCebUE0NTn')
    client.send_message({'from': 'Nexmo', 'to': '+918826897479', 'text': mailtex})
    
    sender = 'maniverma161998@gmail.com'
    receivers = 'divij15103218@gmail.com'
    
    msg = MIMEMultipart()
    msg['From']=sender
    msg['To']=receivers
    msg['Subject']= "someone at the door"
    msg.attach(MIMEText(mailtex,'plain'))
    server = smtplib.SMTP('smtp.gmail.com',25)
    server.starttls()
    server.login(sender,'mani161998')
    text = msg.as_string()
    server.sendmail(sender,receivers,text)
    server.quit()
    '''
    draw_rectangle(img, rect)
    
    draw_text(img, label_text, rect[0], rect[1]-5)
    
    return img

print("Predicting images...")
var=newest('C:/Users/User/Pictures/Camera Roll/')
l=var.split("/")
filepath="C:/Users/User/Pictures/Camera Roll/" + l[5]
test_img1 = cv2.imread(filepath)

predicted_img = predict(test_img1)
print("Prediction complete")

cv2.imshow("Result", cv2.resize(predicted_img, (400, 500)))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()