# Image compression
from pca import *
import matplotlib.image as mplib
import matplotlib.pyplot as plt

def image_compression(filename, n_components = 30):
    # Load and reshape the image
    img = mplib.imread(filename)
    width, height, channels = np.shape(img)
    img = np.reshape(img, (width, height*channels))

    pca = PCA(n_components)
    pca.fit(img)
    
    # Apply PCA to the reshaped image
    img_transformed = pca.transform(img)
    img = pca.inverse_transform(img_transformed)
    img = np.array(np.reshape(img, (width,height,channels)),dtype=float)

    # Normalize pixel intensities
    img = np.array((img - np.min(img)) / (np.max(img) - np.min(img)))
    
    print(f"Explained variance: {pca.explained_variance}")
    return img


def generate_plot(filename,n1,n2,n3):
    fig = plt.figure(figsize=(10,10))
    rows = 2
    columns = 2
    # Upper left subplot
    fig.add_subplot(rows, columns, 1)
    img = mplib.imread(filename)
    plt.imshow(img)
    plt.title("Original picture")
    # Upper right subplot
    fig.add_subplot(rows, columns, 2)
    img = image_compression(filename,n1)
    plt.imshow(img)
    plt.title(f"{n1} principal components")
    # Lower left subplot
    fig.add_subplot(rows, columns, 3)
    img = image_compression(filename,n2)
    plt.imshow(img)
    plt.title(f"{n2} principal components")
    # Lower right subplot
    fig.add_subplot(rows, columns, 4)
    img = image_compression(filename,n3)
    plt.imshow(img)
    plt.title(f"{n3} principal components")
    
    plt.show()

filename = "vangogh.jpg"
n1,n2,n3 = 5, 10, 20
generate_plot(filename, n1,n2,n3)    