# A Simple Canny Edge Detector Implementation (Python)
* [Development of the Canny algorithm](#development-of-the-canny-algorithm)
* [Process of Canny edge detection algorithm](#process-of-canny-edge-detection-algorithm)
	* [Step1: Apply Gaussian filter to smooth the image in order to remove the noise](#apply-gaussian-filter-to-smooth-the-image-in-order-to-remove-the-noise)
	* [Step2: Find the intensity gradients of the image](#find-the-intensity-gradients-of-the-image)
	* [Step3: Apply non-maximum suppression to get rid of spurious response to edge detection](#apply-non-maximum-suppression-to-get-rid-of-spurious-response-to-edge-detection)
	* [Step4: Apply double threshold to determine potential edges](#apply-double-threshold-to-determine-potential-edges)
	* [Step5: Track edge by hysteresis](#track-edge-by-hysteresis)
* [Results And Compare](#compare-with-ground-truth-and-opencv-Bulit-in-algorithm)

# Development of the Canny algorithm 
Canny edge detection is a technique to extract useful structural information from different vision objects and dramatically reduce the amount of data to be processed. It has been widely applied in various computer vision systems. Canny has found that the requirements for the application of edge detection on diverse vision systems are relatively similar. Thus, an edge detection solution to address these requirements can be implemented in a wide range of situations. The general criteria for edge detection include:  
  
* 1. Detection of edge with low error rate, which means that the detection should accurately catch as many edges shown in the image as possible  
* 2. The edge point detected from the operator should accurately localize on the center of the edge. 
* 3. A given edge in the image should only be marked once, and where possible, image noise should not create false edges. 

#Process of Canny edge detection algorithm
***  
```class CannyEdgeDetector(object):

    """
    A naive canny edge detector's implementations:

    Canny Edge Detect Algorithms Procedure:
    Step1: Apply gauss filter to smooth image in order to denoise
    Step2: Generate gradient intensity and phase map for smoothed image
    Step3: Apply non-maximum suppression(NMS) to get rid of spurious response to edge detection(Slim edge)
    Step4: Apply double threshold to determine potentials edges: (strong and weak)
    step5: Track edge by hysteresis:
    Finalize the detection of edges by suppressing all the other edges that are weak and not connected to strong edges.

    see this paper for more details :
    "A Computational Approach to Edge Detection" ,IEEE Transactions on Pattern Analysis and Machine Intelligence ( Volume: PAMI-8 , Issue: 6 , Nov. 1986 )
    """

    def __init__(self,img_path,kernel_size,sigma,threshold):
        """
        :param img_path: /path/to/image
        :param kernel_size: gauss'kernel's size
        :param sigma: gauss kernel's variance
        :param threshold: [low_threshold, high_threshold] for step4 ,eg:[0.2,0.6]

        """
        self.kernel_size = kernel_size
        self.sigma = sigma

        # define a set of kernel
        self.kernel_dic = {
                            'sobel_x': np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
                            'sobel_y': np.array([[1,2,1],[0,0,0],[-1,-2,-1]]),

                          }

        self.threshold = threshold

        # read image with gray mode
        self.img = cv2.imread(img_path,0)

        self.height = self.img.shape[0]
        self.width = self.img.shape[1]


    def gauss_filter(self):
        """
        Step1: Apply gauss filter to smooth image based on given gauss_kernel
        :return: gauss-blur img
        """
        img_blur = cv2.GaussianBlur(self.img,self.kernel_size,self.sigma)
        return img_blur

    def conv(self,img,filter_name):
        """
        implement a simple 2D-Conv operations
        :param filter_name: filter's name
        :return: feature map
        """
        mask = self.kernel_dic[filter_name]
        res = np.zeros((self.height,self.width))

        h_size = mask.shape[0]
        w_size = mask.shape[1]

        for h in range(self.height):
            for w in range(self.width):
                field = img[h:h+h_size, w:w+w_size]
                res[h][w] = np.sum(field*mask)

        return res
```





**The Process of Canny edge detection algorithm can be broken down to 5 different steps:**  
## Apply Gaussian filter to smooth the image in order to remove the noise  

  
  


## Compare With Ground Truth And OpenCV Bulit-in Algorithm
**The Berkeley Segmentation Dataset and Benchmark**  
![original](https://github.com/hfutzzw/CannyEdgeDetector/blob/master/data/circle.jpg) 







