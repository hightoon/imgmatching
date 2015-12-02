#-*- encoding=utf8 -*-

from bottle import get, post, request, run, static_file, route

# OpenCV bindings
import cv2
# To performing path manipulations 
import os
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram 
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
# To read class from file
import csv
# For plotting
import matplotlib.pyplot as plt
# For array manipulations
import numpy as np
# For saving histogram values
from sklearn.externals import joblib
# For command line input
import argparse as ap
# Utility Package
import cvutils


def isimg(impth):
    return impth.endswith('jpg') or impth.endswith('jpeg')

def parse_img():
    X_name, X_test, y_test = joblib.load("lbp.pkl")
    test_images = cvutils.imlist('data/lbp/test/leisi/')
    results_all = {}
    matched = []
    for test_image in test_images:
        if not isimg(test_image):
            continue
        print "\nCalculating Normalized LBP Histogram for {}".format(test_image)
        # Read the image
        im = cv2.imread(test_image)
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        radius = 3
        # Number of points to be considered as neighbourers 
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        x = itemfreq(lbp.ravel())
        # Normalize the histogram
        hist = x[:, 1]/sum(x[:, 1])
        # Display the query image
        results = []
        # For each image in the training dataset
        # Calculate the chi-squared distance and the sort the values
        for index, x in enumerate(X_test):
            score = cv2.compareHist(np.array(x, dtype=np.float32), np.array(hist, dtype=np.float32), cv2.cv.CV_COMP_CHISQR)
            results.append((X_name[index], round(score, 3)))
        results = sorted(results, key=lambda score: score[1])
        results_all[test_image] = results
        print "Displaying scores for {} ** \n".format(test_image)
        for image, score in results:
            if score < 0.012:
                print "{} has score {}".format(image, score)
                matched.append(image)

    return matched

@get('/demo')
def demo():
    return '''
    <html>
        <p>请上传照片</p>
        <form action="/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="data">
            <input type="submit" name="sub">
        </form>
    </html>
    '''

@post('/upload')
def upload():
    data = request.files.data
    with open('data/lbp/test/leisi/uploaded.jpg', 'w') as open_file:
        open_file.write(data.file.read())

    matched_imgs = parse_img()

    content = '<html>'
    for im in matched_imgs:
        content += '<img src=\"/static/%s\" alt="pic">'%im
    content += '</html>'

    return content

@route('/static/<filepath:path>')
def server_static(filepath):
    return static_file(filepath, root='./')

if __name__ == '__main__':
    run(host='0.0.0.0', port=8081, server='cherrypy')

