from datasets.test_dataset import TestDataset
import test
from cosplace_model import cosplace_network as network
import torch
from datetime import datetime
import parser
import os
import commons
import logging
import sys

def load_model(model_path,args):
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(delete_discriminator_layer(model_state_dict))
    return model

def greedy_soup(models_list, args):
    sorted_models = []
    val_ds = TestDataset(args.test_set_folder, queries_folder="queries",
                        positive_dist_threshold=args.positive_dist_threshold)
    print("test set folder ", args.test_set_folder)
    for model in models_list:
        model = model.to(args.device)
        model = model.eval()       
        recalls, _ = test.test(args, val_ds, model)
        sorted_models.append((model, recalls[0]))

    sorted_models.sort(key=compare, reverse=True)
    greedy_soup_ingredients = [sorted_models[0][0]]
    greedy_soup_params = sorted_models[0][0].state_dict()
    num_ingredients = len(greedy_soup_ingredients)
    best_val_rec = sorted_models[0][1]
    for i in range(1,len(sorted_models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }
        new_model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
        new_model.load_state_dict(potential_greedy_soup_params)
        new_model = new_model.to(args.device)
        new_model = new_model.eval()
        new_recall, _ = test.test(args, val_ds, new_model)
        print(f'Potential greedy soup val acc {new_recall[0]}, best so far {best_val_rec}.')
        if new_recall[0] > best_val_rec:
            greedy_soup_ingredients.append(sorted_models[i][0])
            best_val_rec = new_recall[0]
            greedy_soup_params = potential_greedy_soup_params
            print("greedy soup params" , greedy_soup_params)
            print(f'Adding to soup.')
    
    torch.save(greedy_soup_params, f"{args.models_combination}_greedy_soup.pth") 


def uniform_soup(models_list,  args):
    all_params = list(models_list[0].state_dict().keys())
    soup = {}
    for param in all_params:
        soup[param] = sum([m.state_dict()[param].clone() for m in models_list]) / len(models_list)
    agg_model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    agg_model.load_state_dict(soup)
    torch.save(agg_model.state_dict(),f"{args.models_combination}_uniform_soup.pth")
    agg_model = agg_model.to(args.device)
    agg_model = agg_model.eval()

    datasets=["/content/codice/datasets/sf_xs/val", "/content/codice/datasets/sf_xs/test", "/content/codice/datasets/tokyo_xs/test", "/content/codice/datasets/tokyo_night/test"]
    
    for d in datasets:
        val_ds = TestDataset(d, queries_folder="queries", positive_dist_threshold=args.positive_dist_threshold)
        recalls, recalls_str = test.test(args, val_ds, agg_model)
        print(f'{val_ds}: {recalls_str}')


    
def compare(m1):
    return m1[1]

def delete_discriminator_layer(model):
    if "discriminator.1.weight" in model:
            del model["discriminator.1.weight"]
            del model["discriminator.1.bias"]
            del model["discriminator.3.weight"]
            del model["discriminator.3.bias"]
            del model["discriminator.5.weight"]
            del model["discriminator.5.bias"]
    return model


if __name__ == "__main__":

    args = parser.parse_arguments(is_training=False)

    base_path="/content/drive/MyDrive/AML2023/best_model/{}"
    models_directories=["best_model224x224.pth", "best_model_sf_30_1.5.pth", "best_model_af_64_0.5.pth"]
    models_list = []
    for model_path in models_directories:
        m = load_model(base_path.format(model_path),args)
        models_list.append(m)
        

    if args.greedy_soup:
        greedy_soup(models_list, args)
    if args.uniform_soup:
        uniform_soup(models_list, args)