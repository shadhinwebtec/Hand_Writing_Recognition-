from tkinter import *
import cv2
import numpy as np 
from PIL import ImageGrab
from tensorflow import keras 
from keras.models import load_model

model = load_model('Alphabets.h5')
image_folder = 'image/'

root = Tk()
root.resizable(0, 0)
root.title('Handwritten Alphabet Recognition')

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=640, height=480, bg='white')
cv.grid(row=0, column=0, pady=2, sticky=W, columnspan=2)

def clear_widget():
    global cv
    cv.delete('all')

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    cv.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=True, splinesteps=12)
    lastx, lasty = x, y

def activation_event(event):
    global lastx, lasty
    cv.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y

cv.bind('<Button-1>', activation_event)

def recognize_alphanet():
    global image_number
    filename = f'image_{image_number}.png'
    Widget = cv
    x = root.winfo_rootx() + Widget.winfo_rootx()
    y = root.winfo_rooty() + Widget.winfo_rooty()
    x1 = x + Widget.winfo_width()
    y1 = y + Widget.winfo_height()
    
    # Get image and save
    ImageGrab.grab(bbox=(x, y, x1, y1)).save(image_folder + filename)
    image = cv2.imread(image_folder + filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Make a rectangle box around each contour
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        
        # Crop out the alphabet from the image corresponding to the current contour
        alphabet = th[y:y + h, x:x + w]
        
        # Padding the alphabet with 5 pixels of black color (zeros) in each side
        padded_alphabet = np.pad(alphabet, ((5, 5), (5, 5)), 'constant', constant_values=0)
        
        # Resize the padded alphabet to (28, 28)
        resized_alphabet = cv2.resize(padded_alphabet, (28, 28))
        
        # Reshape alphabet to match model input shape
        alphabet_input = resized_alphabet.reshape(1, 28, 28, 1)
        
        # Normalize alphabet
        alphabet_input = alphabet_input / 255.0
        
        # Predict alphabet
        pred = model.predict(alphabet_input)
        final_pred = np.argmax(pred)
        
        # Display predicted alphabet and confidence
        accuracy = int(max(pred[0]) * 100)
        predicted_alphabet = chr(final_pred + 65)  # Convert prediction to alphabet (assuming class indices 0-25)
        prediction_info = f"Predicted Alphabet: {predicted_alphabet}, Confidence: {accuracy}%"
        print(prediction_info)
        
        # Debugging information
        print("Input Shape:", alphabet_input.shape)
        print("Prediction Scores:", pred)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 1
        cv2.putText(image, prediction_info, (x, y - 20), font, fontScale, color, thickness)
    
    # Show the image with predictions
    cv2.imshow('Handwritten Alphabet Recognition', image)
    cv2.waitKey(0)
    
    # Clear the canvas for the next alphabet
    clear_widget()

btn_save = Button(text='Recognize Alphabet', command=recognize_alphanet)
btn_save.grid(row=2, column=0, pady=1, padx=1)
button_clear = Button(text='Clear Alphabet', command=clear_widget)
button_clear.grid(row=2, column=1, pady=1, padx=1)

root.mainloop()
