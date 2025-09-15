function [mdl,pipeopts,results] = loadExperimentItem( filename, options )
%LOADEXPERIMENTITEM Import experiment item from ML-Framework
%
% Syntax:
%   [mdl, pipelineOptions] = loadExperimentItem( filename ) 
%   [mdl, pipelineOptions] = loadExperimentItem( filename, "Pipeline", function ) 
%

% Copyright 2021 The MathWorks Inc.

    arguments
       filename (1,1) string 
       options.Pipeline (1,1) string = ""
    end

    if ~isfile( filename )
        warning( "File not found" )
        mdl = []; pipeopts = []; results = [];
        return
    end
    
    contents = load( filename, 'result' );

    mdl = contents.result.mdl;

    tF = isfield( contents.result.pipesettings, "Options" );
    
    if tF
        
        if options.Pipeline ~= ""
            
            values = validatePipelineParameters( options.Pipeline );
            
            parameters = string(fieldnames( contents.result.pipesettings.Options ));
            fields     = intersect( parameters, values );
            
            validOptions = struct();
            for iField = fields(:)'
                validOptions.(iField)  = contents.result.pipesettings.Options.(iField);
            end %for iField
            
            pipeopts = namedargs2cell( validOptions );
            
        else
            
            pipeopts = namedargs2cell( contents.result.pipesettings.Options );
            
        end %if options.Pipeline
        
    else
        pipeopts = struct();
    end %if ~isempty
    
    results = contents.result;

end %function 

