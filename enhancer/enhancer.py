import cv2
import numpy as np

class Enhancer:
    def __init__(self, method='deconvolution', background_enhancement=True, upscale=1):
        """
        Initialize the Enhancer
        
        Parameters:
        - method: Enhancement method ('deconvolution' or 'sharpen')
        - background_enhancement: Whether to enhance background details
        - upscale: Factor to upscale the image (1 = no upscaling)
        """
        self.method = method
        self.background_enhancement = background_enhancement
        self.upscale = upscale
        
    def enhance(self, image):
        """
        Enhance a blurry image to make it clearer
        
        Parameters:
        - image: Input blurry image (numpy array or file path)
        
        Returns:
        - Enhanced image
        """
        # If image is a file path, load it
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError("Could not read image file")
        
        # Convert to float32 for processing
        image = image.astype(np.float32) / 255.0
        
        # Upscale if needed
        if self.upscale > 1:
            image = cv2.resize(image, None, fx=self.upscale, fy=self.upscale, 
                              interpolation=cv2.INTER_CUBIC)
        
        # Apply enhancement method
        if self.method == 'deconvolution':
            enhanced = self._wiener_deconvolution(image)
        elif self.method == 'sharpen':
            enhanced = self._sharpen_image(image)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Enhance background if needed
        if self.background_enhancement:
            enhanced = self._enhance_background(enhanced)
        
        # Convert back to 8-bit
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def _sharpen_image(self, image):
        """Sharpen the image using a kernel"""
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def _wiener_deconvolution(self, image, kernel_size=15, snr=20):
        """Deconvolution with estimated point spread function"""
        # Estimate PSF (point spread function)
        psf = np.ones((kernel_size, kernel_size, 1)) / (kernel_size ** 2)
        
        # Wiener deconvolution for each channel
        channels = []
        for i in range(3):
            channel = image[:, :, i]
            psf_channel = psf[:, :, 0]
            
            # Compute FFTs
            img_fft = np.fft.fft2(channel)
            psf_fft = np.fft.fft2(psf_channel, s=channel.shape)
            
            # Wiener filter
            kernel = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + 1.0 / snr)
            restored = np.fft.ifft2(img_fft * kernel)
            restored = np.abs(restored)
            channels.append(restored)
        
        # Combine channels
        restored = cv2.merge(channels)
        return restored
    
    def _enhance_background(self, image):
        """Enhance background details using CLAHE"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply((l * 255).astype(np.uint8)) / 255.0
        
        # Merge channels back
        enhanced_lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced
