import cv2
import os

cats = r"C:\Users\Vidhu\Documents\jupyter-notebooks\Cats_and_Dogs\test\cat"
dogs = r"C:\Users\Vidhu\Documents\jupyter-notebooks\Cats_and_Dogs\test\dog"

cat_files = os.listdir(cats)
dog_files = os.listdir(dogs)

for i in cat_files:
    x = cv2.imread(os.path.join(cats, i))
    if x is None:
        print(i)
        os.remove(os.path.join(cats, i))
        print('Fucked up image removed!')
for i in dog_files:
    x = cv2.imread(os.path.join(dogs, i))
    if x is None:    
        print(i)
        os.remove(os.path.join(dogs, i))
        print('Fucked up image removed!')