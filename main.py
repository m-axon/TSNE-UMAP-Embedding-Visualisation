from PIL import Image
import os
import glob
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import torch
import numpy as np
import click
import json


def images_to_sprite(data):
    """
    Creates the sprite image along with any necessary padding
    Source : https://github.com/tensorflow/tensorflow/issues/6322
    Args:
      data: NxHxW[x3] tensor containing the images.
    Returns:
      data: Properly shaped HxWx3 image with any necessary padding.
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


def populate_img_arr(images_paths, size=(100, 100), should_preprocess=False, preprocess_func=lambda x: x):
    """
    Get an array of images for a list of image paths
    Args:
        size: the size of image , in pixels
        should_preprocess: if the images should be processed (according to InceptionV3 requirements)
    Returns:
        arr: An array of the loaded images
    """
    arr = []
    for i, img_path in enumerate(images_paths):
        img = image.load_img(img_path, target_size=size)
        x = image.img_to_array(img)
        arr.append(x)
    arr = np.array(arr)
    if should_preprocess:
        arr = preprocess_func(arr)
    return arr


@click.command()
@click.option('--model', help='pytorch nn.Module')
@click.option('--data', help='Data folder, has to end with /')
@click.option('--name', default="Visualisation", help='Name of visualisation')
@click.option('--sprite_size', default=100, help='Size of sprite')
@click.option('--tensor_name', default="tensor.bytes", help='Name of Tensor file')
@click.option('--sprite_name', default="sprites.png", help='Name of sprites file')
@click.option('--model_input_size', default=299, help='Size of inputs to model-- integer or tuple of two integers')
def main(model, data, name, sprite_size, tensor_name, sprite_name, model_input_size):
    
    if not data.endswith('/'):
        raise ValueError('Makesure --name ends with a "/"')
    
    images_paths = glob.glob(data + "*.jpg")
    images_paths.extend(glob.glob(data + "*.JPG"))
    images_paths.extend(glob.glob(data + "*.png"))

    # model = InceptionV3(include_top=False, pooling='avg')

    img_arr = populate_img_arr(images_paths, size=(model_input_size, model_input_size), should_preprocess=True)
    preds = model.predict(img_arr, batch_size=64)
    preds.tofile("./oss_data/" + tensor_name)

    raw_imgs = populate_img_arr(images_paths, size=(sprite_size, sprite_size), should_preprocess=False)
    sprite = Image.fromarray(images_to_sprite(raw_imgs).astype(np.uint8))
    sprite.save('./oss_data/' + sprite_name)

    oss_json = json.load(open('./oss_data/oss_demo_projector_config.json'))
    tensor_shape = [raw_imgs.shape[0], model.output_shape[1]]
    single_image_dim = [raw_imgs.shape[1], raw_imgs.shape[2]]

    json_to_append = {"tensorName": name,
                      "tensorShape": tensor_shape,
                      "tensorPath": "./oss_data/" + tensor_name,
                      "sprite": {"imagePath": "./oss_data/" + sprite_name,
                                 "singleImageDim": single_image_dim}}
    oss_json['embeddings'].append(json_to_append)
    with open('oss_data/oss_demo_projector_config.json', 'w+') as f:
        json.dump(oss_json, f, ensure_ascii=False, indent=4)


def generate_projector_files(model, dataloader1, dataloader2=None, label_map=None, viz_name='Example_Viz'):
    """
    This helper function generates the needed files for the standalone
    tensorboard projector.
        - oss_demo_projector_config.json
        - embeddings.bytes
        - labels.tsv
        - sprites.png

    Args:
        model (pytorch nn.Module): this is a pytorch model, 
            which implements model.embedding() method, which outputs the chosen
            1D-embedding (pre-classification).
        dataloader1 (pytorch DataLoader object): A pytorch DataLoader object, generating a pre-processed input image 
            and a label.
        dataloader2: if given, will treat as test dataset. Default: None
        labels_txt (list of strings): If provided, assumed to be the same
            length of labels given by the dataloader (and maps respectively to
            ground truth indices). Default: None
        viz_name (string): a name for the vizualization. Default: 'Example_Viz'
    """
    # collect data
    images = []
    embeddings = []
    labels = []

    device = model.device
    for batch, label in dataloader1:
        images.append(batch.to(device))
        embeddings.append(model.embedding(batch))
        if label_map is not None:
            labels.append(('val', label_map[label]))
        else:
            labels.append(('val', label))

    # clip data to max 1000 samples
    images = images[:500]
    embeddings = embeddings[:500]
    labels = labels[:500]

    # ISAR2 test_dataloader outputs 3 elements, where the last one is the name
    # of the file.
    if dataloader2:
        for batch, label, _ in dataloader2:
            images.append(batch)
            embeddings.append(model.embedding(batch))
            if label_map is not None:
                labels.append(('train', label_map[label]))
            else:
                labels.append(('train', label))
    
    # make sure embedding is a 1D representation
    assert embeddings[0].ndim == 2, \
        "model.embedding() should out an (N, d) tensor"
    
    # clip data to max 1000 samples
    images = images[:1000]
    embeddings = embeddings[:1000]
    labels = labels[:1000]

    # stack samples together
    images = torch.stack(images, dim=0).detach().numpy()
    embeddings = torch.stack(embeddings, dim=0).detach().numpy()
    labels = torch.stack(labels, dim=0).detach().tolist()

    ### dump data to files
    # make sure folder exists
    os.makedirs('./oss_data/', exist_ok=True)

    # generate *.bytes file
    embeddings.tofile(f'./oss_data/{viz_name}_embeddings.bytes')

    # generate metadata file
    txt = '\n'.join([
        '\t'.join(map(str, label)) for label in labels
    ])
    with open(f'./oss_data/{viz_name}_labels.tsv', 'w') as f:
        f.write('set\tclass\n')
        f.write(txt)
    f.close()

    # generate sprite image
    # function expects channel last
    image_h, image_w = tuple(images[0].shape[1:])
    sprite = Image.fromarray(images_to_sprite(images.transpose((1, 2, 0))))
    sprite.save(f'./oss_data/{viz_name}_sprite.png')

    # generate config file
    oss_json = json.load(open('./oss_data/oss_demo_projector_config.json'))
    json_to_append = {"tensorName": viz_name,
                      "tensorShape": list(embeddings.shape),
                      "tensorPath": f"./oss_data/{viz_name}_embeddings.bytes",
                      "metadataPath": f"./oss_data/{viz_name}_labels.tsv",
                      "sprite": {"imagePath": f"./oss_data/{viz_name}_sprite.png",
                                 "singleImageDim": [image_h, image_w]}}
    oss_json['embeddings'].append(json_to_append)
    with open('oss_data/oss_demo_projector_config.json', 'w+') as f:
        json.dump(oss_json, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
