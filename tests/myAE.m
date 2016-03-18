

addpath(genpath('C:\Users\chenchen\Documents\GitHub\DeepLearnToolbox'));

dataset = csvread('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\R1.csv');

h1 = [100, 110, 120, 130, 140, 150];
h2 = [20, 30, 40, 50];

% if( isequal(exist('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\AE', 'dir'),7) == 0 )
	
	% mkdir('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\AE');
	
	for a1 = 1:1
		for a2 = 1:1

			mkdir(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\AE\' int2str(h1(a1)) '_' int2str(h2(a2))]);
		
			% Setup and train a stacked denoising autoencoder (SDAE)	
			rand('state',0)
			sae = saesetup([200 h1(a1) h2(a2)]);
			sae.ae{1}.activation_function       = 'sigm';
			sae.ae{1}.learningRate              = 0.25;
			sae.ae{1}.inputZeroMaskedFraction   = 0.5;
			sae.ae{2}.activation_function       = 'sigm';
			sae.ae{2}.learningRate              = 0.25;
			sae.ae{2}.inputZeroMaskedFraction   = 0.5;
			opts.numepochs = 2000;
			opts.batchsize = 864;
			sae = saetrain(sae, dataset, opts);
			hidden_layer1 = sae.ae{1}.a{2};
			hidden_layer2 = sae.ae{2}.a{2};
			hidden_layer1(:,1)=[];
			hidden_layer2(:,1)=[];

			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\AE\' int2str(h1(a1)) '_' int2str(h2(a2)) '\R1_h1.csv'],hidden_layer1);
			csvwrite(['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\AE\' int2str(h1(a1)) '_' int2str(h2(a2)) '\R1_h2.csv'],hidden_layer2);
			
			plot(1:2000,sae.ae{1}.FullBatchError,'-b',1:2000,sae.ae{2}.FullBatchError,'-r');
 			legend(['AE-' int2str(h1(a1))],['AE-' int2str(h2(a2))]);
			title('FullBatchError');
			saveas(gcf,['C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1\AE\' int2str(h1(a1)) '_' int2str(h2(a2)) '\FullBatchError.png']);
	
		end
	end
% end

%  Setup and train a stacked denoising autoencoder (SDAE)
% rand('state',0)
% sae = saesetup([200 100 30]);
% sae.ae{1}.activation_function       = 'sigm';
% sae.ae{1}.learningRate              = 0.25;
% sae.ae{1}.inputZeroMaskedFraction   = 0.5;
% sae.ae{2}.activation_function       = 'sigm';
% sae.ae{2}.learningRate              = 0.25;
% sae.ae{2}.inputZeroMaskedFraction   = 0.5;
% opts.numepochs = 2000;
% opts.batchsize = 864;
% sae = saetrain(sae, dataset, opts);
% hidden_layer = sae.ae{2}.a{2};
% hidden_layer(:,1)=[];

% csvwrite('C:\Users\chenchen\Orientation\WaveleFeatureSheet\R1_AE.csv',hidden_layer);