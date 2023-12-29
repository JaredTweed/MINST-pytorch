import pygame
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from train_model import Net

# Load the trained model
net = Net()
net.load_state_dict(torch.load('mnist_net.pth'))
net.eval()

# Initialize Pygame
pygame.init()
width, height = 560, 560
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("MNIST Digit Recognizer")

# Drawing parameters
drawing = False
last_pos = None
pen_radius = 10
color = (255, 255, 255)  # White
bg_color = (0, 0, 0)  # Black

# Prediction function
def predict_digit(img):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.fromarray(img).convert('L')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = net(image)
        _, predicted = torch.max(output, 1)
        return predicted.item()

# Draw a line from the last position to the current position
def draw_line(screen, start, end, width, color):
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    distance = max(abs(dx), abs(dy))
    for i in range(distance):
        x = int(x1 + float(i) / distance * dx)
        y = int(y1 + float(i) / distance * dy)
        pygame.draw.circle(screen, color, (x, y), width)

# Main loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = pygame.mouse.get_pos()
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.MOUSEMOTION and drawing:
            current_pos = pygame.mouse.get_pos()
            draw_line(screen, last_pos, current_pos, pen_radius, color)
            last_pos = current_pos
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Clear screen
                screen.fill(bg_color)

    pygame.display.flip()
    clock.tick(60)

    # Predict digit
    if not drawing:
        img_data = pygame.surfarray.array3d(screen)
        img_data = np.transpose(img_data, (1, 0, 2))
        prediction = predict_digit(img_data)
        pygame.display.set_caption(f"Prediction: {prediction}, Press 'c' to clear.")

pygame.quit()
