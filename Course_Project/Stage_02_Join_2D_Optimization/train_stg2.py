"""
2. This file is about finetuning the network with 2D Joint Optimization

"""

import options # This module deals with passing arguments from the command line
import utils # This module contains alot of functions that will be used over and over again
from trainer import TrainerStage2 # This module contains the actual code for training the model

if __name__ == "__main__":

    print("=======================================================")
    print("Train structure generator  with joint 2D optimization from novel viewpoints")
    print("=======================================================")

    # Get training inputs from the command line or script file. 
    # Examples of inputs are which GPU used, learning rate, number of epochs, etc...
    cfg = options.get_arguments()

    # Set up folder paths to save various files
    EXPERIMENT = f"{cfg.model}_{cfg.experiment}"
    MODEL_PATH = f"models/{EXPERIMENT}"
    LOG_PATH = f"logs/{EXPERIMENT}"

    utils.make_folder(MODEL_PATH)
    utils.make_folder(LOG_PATH)

    # Set up loss function, data, optimizer, learning rate, etc... for the actual training
    criterions = utils.define_losses()
    dataloaders = utils.make_data_novel(cfg)

    model = utils.build_structure_generator(cfg).to(cfg.device)
    optimizer = utils.make_optimizer(cfg, model)
    scheduler = utils.make_lr_scheduler(cfg, optimizer)

    logger = utils.make_logger(LOG_PATH)
    writer = utils.make_summary_writer(EXPERIMENT)

    # After each epoch of training, log data and save model information
    def on_after_epoch(model, df_hist, images, epoch, saveEpoch):
        utils.save_best_model(MODEL_PATH, model, df_hist)
        utils.checkpoint_model(MODEL_PATH, model, epoch, saveEpoch)
        utils.log_hist(logger, df_hist)
        utils.write_on_board_losses_stg2(writer, df_hist)
        utils.write_on_board_images_stg2(writer, images, epoch)

    # After each batch, record data, and do certain other tasks
    if cfg.lrSched is not None:
        def on_after_batch(iteration):
            utils.write_on_board_lr(writer, scheduler.get_lr(), iteration)
            scheduler.step()
    else: on_after_batch = None

    # Create an instance of the actual training with all the inputs created above.
    # Log the training
    trainer = TrainerStage2(
        cfg, dataloaders, criterions, on_after_epoch, on_after_batch) 

    hist = trainer.train(model, optimizer, scheduler)
    hist.to_csv(f"{LOG_PATH}.csv", index=False)
    writer.close()
