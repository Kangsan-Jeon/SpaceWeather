# SpaceWeather
To expect flare class using CNN with VGG19

Writer : Kang-San Jeon (daiayjks@gmail.com)

## Code
  * [Reference] TF_FLARE.py
    * Reference code of laboratory of Yong-Jae Moon, Univ.Kyunghee, 
    * Use tensorflow
  * main.py
    * train and test code
    * Use keras
  * prepare_data.py
    * To prepare image data to train and test
    * readFits
      > This function read the fits file and show a histogram of image value, image, and clipped image
    * savePngFromFits
      > This function save image to png from fits 
    * convertLabel(with change_datetime)
      > This function convert format of label text file to classify images by flare classes
    * classifyImage
      > This function classify images by flare classes
    * [Flare event label site](https://hesperia.gsfc.nasa.gov/goes/goes_event_listings/)
    
## public_fits_image folder
  * Written by Min-Su Shin
  * Astrophysics, Department of Physics, University of Oxford (2012 - )
  * Department of Astronomy, University of Michigan (2009 - 2012)
  * Department of Astrophysical Sciences, Princeton University (2005 - 2009)
