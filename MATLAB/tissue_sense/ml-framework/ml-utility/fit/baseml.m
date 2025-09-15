classdef baseml
    %BASE Summary of this class goes here
     
    methods (Static)
        
        function [result, scores] = predict( tbl, mdl )
            %PREDICT Predict response for machine learning model
            
            arguments
                tbl table
                mdl
            end
            
            try
                features = tbl( :, mdl.PredictorNames );
            catch
                featuresencoded = baseml.dummyvar( tbl );
                features =  featuresencoded( :, mdl.PredictorNames );
            end
            
            switch class(mdl)
                case 'ClassificationECOC'
                    try
                        [result, ~, ~, scores] = mdl.predict( features );
                    catch
                        [result, ~, ~, scores] = mdl.predict( features.Variables );
                    end
                otherwise
                    try
                        [result, scores] = mdl.predict( features );
                    catch
                        [result, scores] = mdl.predict( features.Variables );
                    end
            end

            
        end %predict

        
        
        function result = dummyvar( tbl )
            %DUMMYVAR Dummy variable coding.
            
            arguments
                tbl table
            end
            
            dummynames = string( baseml.vars( tbl, @(x)~isnumeric(x) & ~isdatetime(x) ) );
            
            for iVar = dummynames(:)'
                
                thisVar = tbl.( iVar );
                
                tF = iscategorical( thisVar );
                
                if tF == false
                    switch class( thisVar )
                        case "char"
                            thisVar = categorical( cellstr( thisVar ) );
                        otherwise
                            thisVar = categorical( thisVar );
                    end
                end
                
                
                dummyvars = dummyvar( thisVar );
                dummyvarnames = iVar + "_" + categories( thisVar );
                dummyvarnames = matlab.lang.makeValidName( dummyvarnames );
                
                tbl = [tbl array2table( dummyvars, 'VariableNames', dummyvarnames )]; %#ok<AGROW>
                tbl = movevars( tbl, dummyvarnames, 'After', iVar );
                
            end
            
            result = removevars( tbl, dummynames );
            
        end %dummyvar
        
        function result = dummyvar2cats( tbl, classes )
            %DUMMYVAR Dummy variable decoding.
            
            arguments
                tbl double
                classes (:,1) string
            end
             
            if size(tbl,1) < size(tbl,2)
                result = (tbl == max(tbl))';
            else
                result = (tbl == max(tbl));
            end
            
            predMat = zeros(length(result),1);
            for ii = 1:length(result)
                row = result(ii,:);
                idx = find(row);
                
                predMat(ii) = idx;
            end
            valset = unique(predMat);         
            result = categorical(predMat,valset,classes(valset));
            
        end %dummyvar
        
        
        function result = partition( tbl, options )
            %PARTITION Create a train/test partition 
            %
            % baseml.partition( data ) create partition with
            % 30% hold-out for test/validation. The parititon is appended
            % to the table as a categorical vector. 
            %
            % baseml.partition( data, "HoldOut", value ) specifies a
            % the value of hold-out (0<value<1) for test/validation 
            %
            
            arguments
                tbl table
                options.HoldOut (1,1) double = .3
            end
            
            args = namedargs2cell( options );
            
            result = tbl;
            
            if options.HoldOut == 1
                thisPartition = repelem( "Test", height(tbl), 1);
            else
                %Partition
                rng( 0 )
                cv = cvpartition( height(tbl), args{:} );
                
                trainIndex  = training(cv);
                testIndex   = test(cv);
                
                thisPartition = strings( height( tbl), 1);
                
                thisPartition(  trainIndex ) = "Train";
                thisPartition(  testIndex ) = "Test";
            end
            
            result.Partition = categorical( thisPartition );
            
        end %partition
        
        function hyperopts = checkHyperparamterOptimizationOptions( hyperopts )
            %CHECKHYPERPARAMETEROPTIMIZATIONOPTIONS Check necessary fields
            % of hyperparameteroptions struct
            %
            % baseml.checkHyperparamterOptimizationOptions( opts ) checks
            % to see if "ShowPlots", "MaxObjectiveEvaluations", and
            % "UseParallel" are defined in hyperparameter optimization
            % options structure

            arguments
                hyperopts (1,1) struct = struct()
            end
            
            if ~isfield(hyperopts, "ShowPlots")
                hyperopts.ShowPlots = false;
            end
            
            if ~isfield(hyperopts, "MaxObjectiveEvaluations")
                hyperopts.MaxObjectiveEvaluations = 30;
            end
            
            if ~isfield(hyperopts, "UseParallel")
                hyperopts.UseParallel = true;
            end    
        end % checkHyperparameterOptimizationOptions
  
    end
    
    methods (Static, Hidden)
        
        function result = defaultfeatures(tbl, rule )      
            arguments
                tbl table
                rule {handleValidation( rule )} = function_handle.empty(0,1);
            end
            
            features   = tbl(:,1:end-1);
            if ~isempty( rule )
                tF     = varfun(rule, features, "OutputFormat", "uni");
                result = baseml.vars( features(:,tF) );
            else
                result = baseml.vars( features );
            end
        end %defaultfeatures 
        
        
        function result = defaultresponse( tbl )
            arguments
                tbl table
            end 
            result  = baseml.vars( tbl(:,end) );    
        end %defaultresponse 
        
        
        function tF = isnumeric( tbl )
            tF = varfun(@isnumeric, tbl, "OutputFormat", "uni" );
        end %isnumeric
        
        
        function result = vars( tbl, rule )
             
            arguments
                tbl table
                rule {handleValidation( rule )} = function_handle.empty(0,1);
            end
            
            if ~isempty( rule )
                tF     = varfun(rule, tbl, "OutputFormat", "uni");
                result = tbl.Properties.VariableNames(tF);
            else
                result = tbl.Properties.VariableNames;
            end
            
        end
         
    end %static, hidden
    
    
end %classdef

function handleValidation(input)
    % Test for specific class
    if ~isa( input, 'function_handle' )
        error('Input must be a function handle or string.')
    end
end 