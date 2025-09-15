classdef aenet 
    %aenet Shallow Net Autoencoder 
    
    methods (Static)
    
        function mdl = fit(tbl, custom, options)
            %autoencoder Fit an autoencoder 

            arguments
               tbl {mustBeClass(tbl, ["table" "timetable"])}
               custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames) 
               custom.Include      (:,1) logical = true( height(tbl), 1);
               custom.HiddenUnit  (1,1) {mustBeInteger(custom.HiddenUnit)} = 10;
               options.EncoderTransferFunction (1,1) {mustBeMember(options.EncoderTransferFunction, ["logsig" "satlin"])} = "logsig"
               options.DecoderTransferFunction (1,1) {mustBeMember(options.DecoderTransferFunction, ["logsig" "satlin" "purelin"])} = "logsig"
               options.MaxEpochs (1,1) {mustBeInteger(options.MaxEpochs)} = 1000;
               options.L2WeightRegularization (1,1) double {mustBePositive(options.L2WeightRegularization)}= 0.001
               options.LossFunction (1,1) {mustBeMember(options.LossFunction, "msesparse")} = "msesparse"
               options.ShowProgressWindow (1,1) logical = true
               options.SparsityProportion (1,1) {mustBeInRange(options.SparsityProportion,[0 1])} = 0.05
               options.SparsityRegularization (1,1) {mustBePositive(options.SparsityRegularization)} = 1;
               options.TrainingAlgorithm (1,1) {mustBeMember(options.TrainingAlgorithm, "trainscg")}
               options.ScaleData (1,1) logical = true
               options.UseGPU (1,1) logical = false
            end

            %Data
            features = tbl( custom.Include, custom.FeatureNames );

            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );
            
            %Train autoencoder for each 
            for iItem = 1:width( featuresencoded )
            
                features = featuresencoded(:,iItem).Variables';

                %Optional Name/Value
                args  = namedargs2cell( options );

                %Autoencoder
                mdl( iItem ) = trainAutoencoder( features, custom.HiddenUnit, args{:} ); %#ok<AGROW>

            end %for iItem
            
        end %function 
        
        
        function value = predict( mdl,tbl, custom )
            %predict 
            
            arguments
               mdl
               tbl  
               custom.FeatureNames (1,:) string = string(tbl.Properties.VariableNames) 
            end
  
            %Data
            features = tbl( :, custom.FeatureNames );

            %Convert Categorical Data
            featuresencoded = baseml.dummyvar( features );

            for iItem = 1:width(featuresencoded)
            
                frames = featuresencoded(:,iItem).Variables';

                prediction = predict( mdl( iItem ), frames );
                
                name = featuresencoded.Properties.VariableNames(iItem)+"Prediction";
    
                tbl.(name) = prediction';
                 
            end %for iItem
             
            %Calculate loss 
            value = aenet.loss( tbl );

        
        end %predict
            
        
        function tbl = anomaly( tbl )
            %ANOMALY 
            
            vars = tbl.Properties.VariableNames;
            tF = ~contains( tbl.Properties.VariableNames, ["Prediction" "Loss", "Anomaly"] );
            
            features = string(vars(tF));
            
            for iFeature = features(:)
            
                tF = isoutlier( tbl.(iFeature+"Loss"), 'mean' );
                tbl.(iFeature+"Anomaly") = tF;
            
            end %for iFeature
            
        end %function
        
        
        function value = sequence2frame( tbl, nWindow, nOverlap, options )
            %SEQUENCE2FRAME
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                nWindow double {mustBeNumeric, mustBePositive, mustBeReal}
                nOverlap double {mustBeNumeric, mustBeReal} = double.empty
                options.InputVariables {validateattributes(options.InputVariables, ...
                    ["cellstr", "char", "string"], "nonempty")} = ""
                options.GroupingVariables {validateattributes(options.GroupingVariables, ...
                    ["cellstr", "char", "string"], "nonempty")} = ""
            end
            
            args = namedargs2cell( options );
            
            value = util.sequence2frame( tbl, nWindow, nOverlap, args{:});

            value = addprop(value, "nWindow", "table");
            value.Properties.CustomProperties.nWindow = nWindow;
            
            
        end %function
            
        
        function value = frame2sequence( tbl, options )
            %FRAME2SEQUENCE
            
            arguments
                tbl {mustBeClass(tbl, ["table" "timetable"])}
                 options.InputVariables {validateattributes(options.InputVariables, ...
                     ["cellstr", "char", "string"], "nonempty")} = ""
            end
            
            if options.InputVariables ~= ""
                
                if util.isvar(tbl, options.InputVariables)
                   tbl = tbl(:,options.InputVariables);
                end
                
            end
            
            nWindow = tbl.Properties.CustomProperties.nWindow;
          
            sequences = vertcat( ...
                nan(nWindow, width(tbl)),...
                varfun(@(x)x(:,end), tbl(2:end,:) ).Variables );
            
            value = array2table(sequences, ...
                'VariableNames', tbl.Properties.VariableNames);
            
            
        end %function
        
    end %methods

    methods (Static, Access='private')
        
        function tbl = loss(tbl)
            %LOSS
            
            vars = tbl.Properties.VariableNames;
            tF = ~contains( tbl.Properties.VariableNames, ["Prediction","Loss"] );
            
            features = string(vars(tF));
            
            for iFeature = features(:)
                
                tbl.(iFeature+"Loss") = rowfun(@(y,x)sqrt(sum((y-x).^2)), tbl, ...
                    "InputVariables", [iFeature iFeature+"Prediction"], ...
                    "OutputFormat", "uniform");
                
            end %for iFeature
            
        end %function
        
    end %methods 
    
    
end %classdef 


%Validation function
function mustBeInRange(a,b)
   if any(a(:) < b(1)) || any(a(:) > b(2))
      error(['Value assigned to Data property is not in range ',...
         num2str(b(1)),'...',num2str(b(2))])
   end
end


function mustBeClass( value, members )

    if ~ismember(class( value ), members)

        msg = repelem("%s",1,numel(members) );
        if numel(members) > 1
            msg = join(msg(1:end-1), ",") + ", or " + msg(end);
        end

        throwAsCaller( MException("util:mustBeClass", ...
            sprintf( 'Value must be of the following types: '+msg, members) ))
    end

end %function




% You can use this code to add stride 

% resultOfBuffer.Properties.CustomProperties.Index = map;
% resultOfBuffer.Properties.CustomProperties.nSample = height(tbl);
% resultOfBuffer = addprop(resultOfBuffer, "Index", "table" );
% resultOfBuffer = addprop(resultOfBuffer, "nSample", "table" );
%
%             for iItem = 1:width(featuresencoded)
%
%                [nSample,nFrame] = size(featuresencoded(:,iItem).Variables);
%
%                 frames = featuresencoded(:,iItem).Variables;
%
%                 code = nan(nSample, nFrame );
%                 for iFrame = 1:size( nFrame )
%
%                     frame = frames(:,iFrame);
%                     code(:, iFrame) = mdl(iItem).encode( frame );
%
%                 end
%
%                 features.Encode = code;
%
%             end %for iItem

%             nSample = result.Properties.CustomProperties.nSample;
%             index = result.Properties.CustomProperties.Index;
%
%             [indexInSequence,indexInFrame]=unique(index);
%             maxValue = max(index, [], 'all');
%
%             toDrop = ( indexInFrame > maxValue );
%
%             indexInSequence(toDrop) = [];
%             indexInFrame(toDrop) = [];
%
%             indexLoss = index(:,end);
%
%             sequence= nan( nSample,width(result) );
%             for iVar = 1:width(result)
%
%                 var = result(:,iVar).Variables;
%
%                 if size(var,2) > 2
%                     sequence(indexInSequence,iVar) = var(indexInFrame);
%                 else
%                     sequence(indexLoss,iVar) = var;
%                 end
%
%             end
%
%             result = array2table(sequence, ...
%                 'VariableNames', result.Properties.VariableNames);

