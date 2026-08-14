import argparse
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description="Gera um grid com as etapas de pre-processamento.")
    parser.add_argument('--image', required=True, help='Caminho para a imagem de entrada')
    parser.add_argument('--output', default='pipeline_grid.png', help='Caminho para salvar o grid (ex: grid.png)')
    args = parser.parse_args()

    # 1. Carregar imagem original
    original = Image.open(args.image).convert('RGB')
    
    # 2. Resize (Tamanho da rede)
    resize_transform = transforms.Resize((224, 224))
    resized = resize_transform(original)
    
    # 3. Grayscale (3 canais)
    gray_transform = transforms.Grayscale(num_output_channels=3)
    grayscale = gray_transform(resized)
    
    # --- Plotando o Grid ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('1. Original')
    axes[0].axis('off')
    
    axes[1].imshow(resized)
    axes[1].set_title('2. Resized (224x224)')
    axes[1].axis('off')
    
    axes[2].imshow(grayscale)
    axes[2].set_title('3. Grayscale')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Grid gerado com sucesso e salvo em: {args.output}")

if __name__ == '__main__':
    main()
