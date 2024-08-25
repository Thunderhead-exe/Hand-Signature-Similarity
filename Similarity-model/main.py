from similarity_model import *

# Load model
model = SiameseModel()
model.load_state_dict(torch.load('/Users/ademsalehbk/Desktop/Spark-it/Signature_Similarity/similarity-model/convnet_best_loss.pt', map_location=torch.device('cpu'))['model'])

#CFG.projection2d=True
model.eval()

# Load test images
img1 = preprocess.load_signature("/Users/ademsalehbk/Desktop/Spark-it/Signature_Similarity/test-images/05_065.png") # Original
img2 = preprocess.load_signature("/Users/ademsalehbk/Desktop/Spark-it/Signature_Similarity/test-images/05_065_forged.png") # Forged

img1 = preprocess.preprocess_signature(img1, Config.canvas_size, Config.input_size)
img2 = preprocess.preprocess_signature(img2, Config.canvas_size, Config.input_size)

img1 = torch.from_numpy(img1)
img2 = torch.from_numpy(img2)

concatenated = torch.cat((img1, img2),1)
with torch.no_grad():
    op1, op2, confidence = model(img1, img2)

# Output Metrics
confidence = confidence.sigmoid().detach().to('cpu')
cos_sim = F.cosine_similarity(op1, op2)

# Show result
preprocess.imshow(torchvision.utils.make_grid(concatenated.unsqueeze(1)), f'similarity: {cos_sim.item():.2f} Confidence: {confidence.item():.2f}')
#plt.savefig('siamese.png')

print('conf', confidence, ' cos_sim', cos_sim)

