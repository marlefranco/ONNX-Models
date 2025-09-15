function [mdl,pipeopts] = prepareInference( item, options )
%PREPAREINFERENCE Import experiment item from ML-Framework
%
% Syntax:
%   [mdl, pipelineOptions] = loadExperimentItem( filename ) 
%   [mdl, pipelineOptions] = loadExperimentItem( filename, "Pipeline", function ) 
%

    arguments
       item (1,1) struct
       options.Pipeline (1,1) string 
    end

    mdl = item.mdl;

    tF = isfield( item.pipesettings, "Options" );
    
    if tF
        
        if options.Pipeline ~= ""
            
            values = validatePipelineParameters( options.Pipeline );
            
            parameters = string(fieldnames( item.pipesettings.Options ));
            fields     = intersect( parameters, values );
            
            validOptions = struct();
            for iField = fields(:)'
                validOptions.(iField)  = item.pipesettings.Options.(iField);
            end %for iField
            
            pipeopts = namedargs2cell( validOptions );
            
        else
            
            pipeopts = namedargs2cell( item.pipesettings.Options );
            
        end %if options.Pipeline
        
    else
        pipeopts = struct();
    end %if ~isempty

end %function 

