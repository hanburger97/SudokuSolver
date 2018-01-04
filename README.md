Our project is for use on a Rasberry Pi. It will consist of the user pressing a button connected to a rPi, the Pi will then snap a picture of handwritten/printed C code with a connected camera module. It will then submit that picture to the Google Vision API to parse it to text. Because the code needs to be as accurate as possible to run, multiple filters will then be applied to the text, and then all possible interpretations of the erronous code will be tested until one runs. Once once of the code interpretations runs successfully, it will output the C output to a display connected to the rPi.

