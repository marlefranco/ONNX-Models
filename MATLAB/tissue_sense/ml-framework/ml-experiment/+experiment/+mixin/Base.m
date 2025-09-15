classdef Base < matlab.mixin.SetGet
    %experiment.mixin.Base 
    %  
    %
    
    methods (Access = protected)
    
        function validLearners(  obj )   %#ok<MANU>
        end %function
        
        function updateLearners(  obj )    %#ok<MANU>
        end %function
        
        function value = initializeModelParams( obj, trial, mthd )
            %initializeModelParams
            
            arguments
                obj
                trial experiment.item.Base
                mthd (1,1) string {mustBeMember(mthd, ["automl", "selectml", "stackml", "fit"])}
            end
            
            %Initialize item model parameters
            if obj.Type == "PDM" 
                value = trial.initializeModelParams( "pdm" );
            else
                value = trial.initializeModelParams();
            end %if obj.Type
                        
            %Parse settings for learner property (extract then remove)
            switch mthd
                case "automl"
                    [value, learners] = parseAndExtractArgs( value, "Learners" );
                    
                    try
                        %If Model configuration exists, else use all valid learners
                        if ~isempty( value )
                            
                            %If Learners are specified, else use all valid
                            %learners
                            if ~isempty( learners )
                                
                                if any( matches( learners, "all") )
                                    obj.validLearners( trial );
                                else
                                    obj.validLearners( trial, learners);
                                end %if any( matches...
                                
                            else
                                obj.validLearners( trial );
                            end %if ~isempty( learners)
                        else
                            obj.validLearners( trial );
                        end %if ~isempty( value )
                        
                    catch ME
                        throw(ME)
                    end
                    
                case "selectml"
                    
                    learners = parseArgs( value, "Learners" );
                    
                    if ~isempty( learners )
                        obj.updateLearners( learners );
                    else
                        switch obj.Type
                            case "Classification"
                                obj.updateLearners( "auto" );
                            otherwise
                                obj.updateLearners( "all" );
                        end %switch 
                        
                    end %if ~isempty( learners)
                    
                case "stackml"
                    
                    %Override defaults (only applies to metamodel). No
                    %optimization on stacked model.
                    value = struct();
                    
                    %Use prediction as features, invert training set
                    vars     = trial.VariableNames;
                    response = trial.response;
                    features = vars(contains(vars, "Prediction"));
                    
                    value.PredictorNames = features;
                    value.ResponseName = response;
                    value.Include = ~trial.trainingobs;
                    value.Learners = obj.Learners;
                            
                    value = namedargs2cell( value );
                      
                case "fit"
                    
                    %If learners is supplied to standard fit* function, remove
                    value = parseAndExtractArgs( value, "Learners" );
                    obj.updateLearners( "n/a" );
                    
                otherwise
                    error("Unhandled option.")
            end %switch
                        
        end %function
        
    end %methods
    
end %classdef 


%Local
function [value, splitvalue] = parseAndExtractArgs( value, name )
    %parseAndExtractArgs

    indx = cellfun(@(x)ismember(x,name), value(1:2:end) );

    if ~isempty( indx ) &&  any( indx )
        splitvalue = value{ find(indx)*2 };
        value( find(indx)*2-1:find(indx)*2 ) = [];
    else
        splitvalue = {};
    end %if ~isempty(indx)

end %function


function splitvalue = parseArgs( value, name )
    %parseArgs
    
    indx = cellfun(@(x)ismember(x,name), value(1:2:end) );

    if ~isempty(indx) &&  any( indx )
        splitvalue = value{ find(indx)*2 };
    else
        splitvalue = {};
    end %if ~isempty(indx)

end %function