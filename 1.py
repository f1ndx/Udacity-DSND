### TODO: Write your algorithm.

def make_prediction(img_path, model):
    _, ax = plt.subplots()
    if dog_detector(img_path):
        print("Hey look! It's a DOG!")

        # display image
        img = mpimg.imread(img_path)
        _ = ax.imshow(img)
        plt.axis('off')
        plt.show()

        # display breeds and probabilities
        indices, breeds, probs = inception_predict_breed(img_path, model)
        predictions = ""
        for i, breed, prob in zip(indices, breeds, probs):
            predictions += f"\n- {breed} ({prob:.2})"
        print(f"The breed appears to be:{predictions}")

        # display sample of matching breed images
        fig = plt.figure(figsize=(15,4)) 
        for i in enumerate(indices):
            img = mpimg.imread(test_image_lookup[i[1]][2])
            ax = fig.add_subplot(1,3,i[0]+1)
            ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
            plt.title(test_image_lookup[i[1]][1])
            plt.axis('off')
        plt.show()

    elif face_detector(img_path):
        print("Hey look! It's a HUMAN!")

        # display image
        img = mpimg.imread(img_path)
        _ = ax.imshow(img)        
        plt.axis('off')
        plt.show()

        # display breeds and probabilities
        indices, breeds, probs = inception_predict_breed(img_path, model)
        resemblance = ""
        for i, breed, prob in zip(indices, breeds, probs):
            resemblance += f"\n- {breed} ({prob:.2})"
        print(f"The person in this image resembles these breeds:{resemblance}")

        # display sample of matching breed images
        fig = plt.figure(figsize=(15,4)) 
        for i in enumerate(indices):
            img = mpimg.imread(test_image_lookup[i[1]][2])
            ax = fig.add_subplot(1,3,i[0]+1)
            ax.imshow(img.squeeze(), cmap="gray", interpolation='nearest')
            plt.title(test_image_lookup[i[1]][1])
            plt.axis('off')
        plt.show()        

    else:
        print("I'm sorry Dave, I'm afraid I can't do that...¯\_(ツ)_/¯. Please try another image.")
        img = mpimg.imread(img_path)
        _ = ax.imshow(img)        
        plt.axis('off')
        plt.show()
        
    #K.clear_session()
    print("\n"*3)