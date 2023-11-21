"""
 
 1. This file is about the first stage of training. This is where we create the autoencoder and train it on point cloud 3D models to teach the network how to decode and encode a particular 3D object. Note that this model is good for training only one type of model at a time.


"""


import options # This module deals with passing arguments from the command line
import utils # This module contains alot of functions that will be used over and over again
from trainer import TrainerStage1 # This module contains the actual code for training the model


if __name__ == "__main__":

    print("=======================================================")
    print("Pretrain structure generator with fixed viewpoints")
    print("=======================================================")

    # Get training inputs from the command line or script file. 
    # Examples of inputs are which GPU used, learning rate, number of epochs, etc...
    cfg = options.get_arguments() 


    # Set up folder paths to save various files
    EXPERIMENT = f"{cfg.model}_{cfg.experiment}"
    MODEL_PATH = f"models/{EXPERIMENT}"
    LOG_PATH = f"logs/{EXPERIMENT}"

    utils.make_folder(MODEL_PATH)
    utils.make_folder(MODEL_PATH)

    # Set up loss function, data, optimizer, learning rate, etc... for the actual training
    criterions = utils.define_losses()
    dataloaders = utils.make_data_fixed(cfg)

    model = utils.build_structure_generator(cfg).to(cfg.device)
    optimizer = utils.make_optimizer(cfg, model)
    scheduler = utils.make_lr_scheduler(cfg, optimizer)

    logger = utils.make_logger(LOG_PATH)
    writer = utils.make_summary_writer(EXPERIMENT)
    
    # After each epoch of training, log data and save model information
    def on_after_epoch(model, df_hist, images, epoch):
        utils.save_best_model(MODEL_PATH, model, df_hist)
        utils.log_hist(logger, df_hist)
        utils.write_on_board_losses_stg1(writer, df_hist)
        utils.write_on_board_images_stg1(writer, images, epoch)


    # After each batch, record data, and do certain other tasks
    if cfg.lrSched is not None:
        def on_after_batch(iteration):
            utils.write_on_board_lr(writer, scheduler.get_lr(), iteration)
            scheduler.step(iteration)
    else: on_after_batch = None

    # Create an instance of the actual training with all the inputs created above.
    # Log the training
    trainer = TrainerStage1(
        cfg, dataloaders, criterions, on_after_epoch, on_after_batch)

    hist = trainer.train(model, optimizer, scheduler)
    hist.to_csv(f"{LOG_PATH}.csv", index=False)
    writer.close()
