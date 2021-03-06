

	addpath(genpath('C:\Users\chenchen\Documents\GitHub\DeepLearnToolbox'));

	dataset = csvread('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\R2.csv');

	h1 = [300, 350, 400, 450, 500, 550, 600, 900, 1200, 1500, 1800];
	h2 = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
	
	if( isequal(exist('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred', 'dir'),7) == 0 )
	
		mkdir('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred');
	
	end
	
	for a1 = 1:size(h1,2)
		for a2 = 1:size(h2,2)

			mkdir(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2))]);
		
			% Setup and train a stacked denoising autoencoder (SDAE)	
			rand('state',0)
			sae = saesetup([200 h1(a1) h2(a2) h1(a1)]);
			
			for n = 1:3
			
				sae.ae{n}.activation_function       = 'sigm';
				sae.ae{n}.learningRate              = 0.25;
				sae.ae{n}.inputZeroMaskedFraction   = 0;
			
			end
			
			opts.numepochs = 1000;
			opts.batchsize = 24;
			sae = saetrain(sae, dataset, opts);
			
			% SAE FinHundrede-tuning
            nn = nnsetup([200, h1(a1), h2(a2), h1(a1), 200]);    %h3               
            nn.activation_function              = 'sigm';
            nn.learningRate                     = 0.25;
            nn.inputZeroMaskedFraction          = 0;
			for n = 1:3
			
				nn.W{n} = sae.ae{n}.W{1}; %every ae just 1 hidden layer, so W{1}
				
			end
			
			% FinHundrede-tuning之設定
			opts.numepochs = 1000;
			opts.batchsize = 24;
			
			% 開始訓練
			[nn, L, Fine_tune_error] = nntrain(nn, dataset, dataset, opts); 
			
			% total_batch
			nn = nnff(nn, dataset, dataset);
			
			%印出權重可加在這邊 start
			input_layer = nn.a{1};
			input_layer(:,1)=[];
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\input_layer.csv'],input_layer);
			
			hidden1_layer = nn.a{2};
			hidden1_layer(:,1)=[];
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\hidden1_layer.csv'],hidden1_layer);
			
			hidden2_layer = nn.a{3};
			hidden2_layer(:,1)=[];
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\hidden2_layer.csv'],hidden2_layer);
			
			hidden3_layer = nn.a{4};
			hidden3_layer(:,1)=[];
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\hidden3_layer.csv'],hidden3_layer);
			
			output_layer = nn.a{5};
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\output_layer.csv'],output_layer);
			%印出權重可加在這邊 end
			
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\1stLayerMSE.csv'],sae.ae{1}.FullBatchError);
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\2ndLayerMSE.csv'],sae.ae{2}.FullBatchError);
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\3rdLayerMSE.csv'],sae.ae{3}.FullBatchError);
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\FineTuneMSE.csv'],Fine_tune_error);
			
			%test
			[er, bad] = nntest(nn, dataset, dataset);
            disp(['er= ',num2str(er)]);    
			
			% 畫Error-rate
			plot(1:1000,sae.ae{1}.FullBatchError,'-b',1:1000,sae.ae{2}.FullBatchError,'-r',1:1000,sae.ae{3}.FullBatchError,'-g',1:1000,Fine_tune_error,'-m');
 			legend(['AE-' int2str(h1(a1))],['AE-' int2str(h2(a2))],['AE-' int2str(h1(a1))],['Fine-tune-error']);
			title('MSE-fullBatchError');
			saveas(gcf,['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R2\AE\FinHundred\' int2str(h1(a1)) '_' int2str(h2(a2)) '\MSE-FullBatchError.png']);
	
		end
	end
