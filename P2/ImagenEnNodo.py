from PIL import Image 

  
def main(): 
    try: 
         #Relative Path 
        img2 = Image.open("hijo.jpg") 
        width, height = img2.size 
   
        img2 = img2.resize((width*2, height*2)) 
          
        #Saved in the same relative location 
        img2.save("hijo2.jpg")  
    except IOError: 
        pass
  
if __name__ == "__main__": 
    main()
    
    
def main(): 
    try: 
        #Relative Path 
        #Image on which we want to paste 
        img = Image.open("grafo4.jpg")  
          
        #Relative Path 
        #Image which we want to paste 
        img1 = Image.open("abuelo2.jpg")
        img2 = Image.open("abuela2.jpg")
        img3 = Image.open("marge2.jpg")
        img4 = Image.open("hermana2.jpg")
        img5 = Image.open("hijo2.jpg")
        img.paste(img1, (2200,1250)) 
        img.paste(img2, (1200,1500))
        img.paste(img3, (3200,3100)) 
        img.paste(img4, (100,0))
        img.paste(img5, (5200,3000)) 
        
        #Saved in the same relative location 
        img.save("pasted_picture.jpg") 
                  
    except IOError: 
        pass
  
if __name__ == "__main__": 
    main()