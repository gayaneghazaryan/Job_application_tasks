"""
Image Palette Extraction Script

This script demonstrates the extraction of a color palette from an image using the K-means clustering algorithm.
It provides a function, extract_palette, which takes an image URL, palette size, and optional random state as inputs,
and displays the color palette using matplotlib.

"""
import requests
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def extract_palette(img_url, palette_size, random_state = None):
    """
    Extracts a color palette from an image using the K-means clustering algorithm.
    
    Parameters:
    - img_url (string): The URL of the image from which the color palette will be extracted.
    - palette_size (integer): The number of colors to be included in the palette.
    - random_state (integer or None, optional): The random number generation for centroid initialization. 
      If None, the initialization will be random. Defaults to None.
      
    """
    
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=palette_size, random_state=random_state)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_

    plt.imshow([colors.astype(int)])
    plt.axis('off')
    plt.show()

