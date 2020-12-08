from scipy import stats
import torchvision


def accuracy(dataloaders_x,model,BATCH_SIZE=64):
    total=0
    count=len(dataloaders_x['test'])
    image=[]
    generated_image=[]
    actual=[]
    predicted=[]
    results=[]
    for i, batch in enumerate(dataloaders_x['test']):
        if i==count-1:
            break

        inputs, classes=batch[0],batch[1]
        
        inputs=inputs.to(device)
        generated=cgan.inference(inputs)    
        for j in range(64):
            m=stats.mode([generated[0][j],generated[1][j],generated[2][j],generated[3][j],generated[4][j]])
            if (classes[j]==torch.tensor(m[0])):
                total+=1
            if i==1:
                predicted.append((int(m[0])))
                
        #datapoints collection to plot
        if i==1:
            image.append(inputs)
            generated_image.append(generated[0])
            actual.append(classes)            
        

    accuracy=total/(count*BATCH_SIZE)   
    return accuracy,image[0],generated_image[0],np.array(actual[0]),predicted


accuracy,images, generated_images,actual, predicted=accuracy(dataloaders,cgan)# real
