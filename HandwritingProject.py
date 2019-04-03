import matplotlib.pyplot as plt
from sklearn import datasets, svm
digits= datasets.load_digits()
print "digits :", digits.keys()
print "digits target ", digits.target

image_and_labels = list(zip(digits.images, digits.target))
print "len(image_and_labels) :", len(image_and_labels)

#for training the data
    #0       img0   0
for index, [image, label] in enumerate(image_and_labels[: 5]): # enumerate only 5 values
    print "index :", index, "image: \n", image, "label : ", label
    plt.subplot(2, 5, index+1)
    plt.axis('on') #for ticks
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')#show the image, gray_r= to convert black n white
    plt.title('training %i' %label)

#we reshape the data e,g, 8X8 matrix to flat data
n_samples = len(digits.images)
print "n_samples=",n_samples
image_data = digits.images.reshape((n_samples, -1)) #resize to flat data #-1 means reduce dimension by one
print "after reshape len(imagedata[0])", len(image_data[0])
classifier =svm.SVC(gamma=0.001) #gamma= learning rate
classifier.fit(image_data[:n_samples//2], digits.target[:n_samples//2])
expected = digits.target[n_samples//2:]
predicted = classifier.predict(image_data[n_samples//2:])

images_an_prediction = list(zip(digits.images[n_samples//2:], predicted)) #predicted image
for index,[image,predicted] in enumerate(images_an_prediction[:5]):
    plt.subplot(2, 5, index + 6)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('prediction %i' %predicted)

print "original value :", digits.target[n_samples//2:(n_samples//2)+5] #2nd half+5

plt.show()

#Pillow library
from scipy.misc import imread,imresize,bytescale
img = imread("Two.jpeg")
img = imresize(img, (8, 8))
img = img.astype(digits.images.dtype)

img = bytescale(img, high= 16.0, low=0) #to set the resolution

print "img :", img

x_testdata= []

for c in img:
    for r in c:
        x_testdata.append(sum(r)/3.0)
print "x_teatdatan", x_testdata
x_testdata= [x_testdata] #50 % test data, [#] one image
print "len(x_trstdata)", len(x_testdata) #no of image
print "machine_output", classifier.predict(x_testdata)

plt.show()
