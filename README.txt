Automatic Stock Trading Bot README:
This is a proof of concept project which uses an artificial neural network to perform trades on
Alpaca. https://alpaca.markets/ the application follows an overnight hold trade strategy; selling in the morning 
and placing orders to buy those stocks with the highest estimated gains from today to tommorow during the afternoon.
This project was developed for and tested on alpaca's paper trading environment only.

Project Manifest:
	-Network
	-AutoTrader.py
	-config.txt
	-README.txt
	-scaler.pkl
	-TradeDriver.py
	-universe

Installation instructions:
	To install this application place all files contained in this directory into the following directory:

		C:\Program Files\TradeBot\

	This application requires the following in order to run:
		A Python interpreter must be installed. 
			You can download the latest version of python here: https://www.python.org/downloads/
		You must have an Alpaca Paper Trading Account, and the associated keys in order to make use of this application.
			sign up for an account by following this link: https://app.alpaca.markets/signup
			enter your email address and choose a password for your alpaca account
			you will be redirected to a page which asks for a verification code.
			check your email for a verification email from Alpaca and enter that code on the verification page, then click verify.
			you will then be redirected to your alpaca home page, click on "Go to Paper Account" at the top left of the page.
			you can obtain your Paper API Keys by clicking on the "View" button on the right hand side of the page then clicking on the "Generate New Key" button.
			copy your API Key ID and Secret Key into the config file next to the appropriate label to the right of the ':'
			 
		The following python packages be installed:
			Keras
			Tensorflow
			sikit-learn
			Alpaca Trade API
			Numpy
			Pandas

		to install these packages, open command prompt from the start menu and run the following command:

	 		pip install numpy==1.19.3 pandas sklearn tensorflow keras alpaca-trade-api

		Once the command has finished you have completed the installation.

Running the application:

	Navigate to the folder in which you have installed the script files by running the following command:

		cd %SystemDrive%\Program Files\TradeBot\

	If you've chosen to install this application in a different location than the recommended one, replace "%SystemDrive%\Program Files\" with the file path to that location instead.
	Start the application running using the following command.

		 pythonw TradeDriver.py
	
	You may then close the active console window, the application will be running in the background.	