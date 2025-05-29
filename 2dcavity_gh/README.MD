To run the code ensure the following liubraries are installed

    pip install numpy==2.2.5
    pip install pandas==2.2.3
    pip install matplotlib==3.10.1
    pip install seaborn==0.13.2
    pip install scikit-learn==1.6.1
    pip install scipy==1.15.2
    pip install jupyterlab==4.3.5 
    pip install ipython==9.0.1 
    pip install tqdm==4.67.1 
    pip install requests==2.32.3 
    pip install h5py==3.13.0 
    pip install pyyaml==6.0.2
    pip install opencv-python==4.11.0.86
    pip install transformers==4.49.0 
    pip install datasets==3.3.2
    pip install scikit-image==0.25.2
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install accelerate==1.4.0
    pip install diffusers["torch"]==0.32.2 transformers==4.49.0
    pip install h5py==3.13.0
    pip install lightning==2.5.0.post0
    pip install tensorboard==2.19.0
    pip install ngsolve==6.2.2501
    

To generate the training and validation data, simply run
	python3 utils_cavity_data.py
	
To train the model, simply run
	python3 pinn_cavity.py --config config_cavity.yaml > output.txt 
	

Overview of files:
	utils_cavity_data.py contains the code to generate the trainig and validation data.
	utils_cavity_ngsolve.py contains the code to run NGSolve i.e. the finite element solver used to train NewPINNs
	utils_cavity_plots.py contains code to generate various plots.
	pinn_cavity.py contains the code to train the U-Net using the NewPINNs method
	config_cavity.yaml contains the hyperparameters for the training procedure.
	

