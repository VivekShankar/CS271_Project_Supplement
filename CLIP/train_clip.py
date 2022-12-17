def caption_to_str(caption_row, LABELS):
  only_labels_caption_row = caption_row[LABELS]
  only_labels_caption_row = only_labels_caption_row[only_labels_caption_row == 1]
  captions = only_labels_caption_row.keys().to_numpy()
  caption_str = "A chest X-ray of " + " ".join(captions)
  return caption_str

# https://github.com/elsevierlabs-os/clip-image-search/blob/main/fine-tuning/train.py#L105
def do_train(df, get_image_func, LABELS, model, processor, optimizer, lr_scheduler, device):  
  train_loss = 0
  bsz = 4
  model.train()
  model.to(device)
  batch = {
    "image": [],
    "caption": [],
  }
  num_training_steps = 100
  loss_lst = []
  for bid, row in df.iterrows():
    if(bid == num_training_steps):
      break
    image_id = row["image_id"]

    try:
      image = get_image_func(image_id)
      batch["image"].append(image)
      caption_str = caption_to_str(row, LABELS)
      batch["caption"].append(caption_str)
    except:
      print("image errored")

    if((bid+1) % bsz == 0):
      print(batch["caption"])
      inputs = processor(text=batch["caption"], images=batch["image"], return_tensors="pt", padding=True)
      inputs = inputs.to(device)
      # if bid % 1 == 0:
      #     print("...{:d} training steps complete".format(bid))
      # batch = {k: v.to(device) for k, v in batch.items()}
      outputs = model(**inputs, return_loss=True)
      print(outputs.logits_per_text)
      # print("OUTPUTS:", outputs)
      loss = outputs.loss
      print("LOSS:", loss)
      loss_lst.append(loss)
      train_loss += loss.detach().cpu().numpy()
      loss.backward()
      
      optimizer.step()
      lr_scheduler.step()
      optimizer.zero_grad()

      batch = {
        "image": [],
        "caption": [],
      }

  print("...{:d} training steps COMPLETE".format(bid))
  return loss_lst

def setup_train(model):
  # setup model optimization stuff
  from transformers import (
      CLIPModel, CLIPProcessor,
      AdamW, get_scheduler
  )

  init_lr = 1e-3
  optimizer = AdamW(model.parameters(), 
                    lr=init_lr)

  num_epochs = 3
  num_training_steps = 1000

  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  return optimizer, lr_scheduler

def main(df, get_image_func, CLASS_TEXT, model, processor, device):
  optimizer, lr_scheduler = setup_train(model)

  loss_lst = do_train(df, get_image_func, CLASS_TEXT, model, processor, optimizer, lr_scheduler, device)
  return loss_lst

