function result = validatePipelineParameters( functionname )
    %validatePipelineParameters

    % Copyright 2021 The MathWorks Inc.

    location = which( functionname );
    
    %Read function content 
    contents = fileread( location );
    
    %Check to see if ".m" is included in functionname
    functioncheck = extractBefore(functionname, ".m");
    if ~ismissing(functioncheck)
        functionname = functioncheck;
    end
    
    %Special case: package function
    if contains(functionname, ".")
        parts = strsplit( functionname, '.' );
        functionname = string( parts(end) );
    end
    
    %Extract contents after function signature
    parts    = regexp( contents,...
        "function[^=\n\r]*={1}?\s*"+functionname+"\s*(", "start" );
    
    if isempty(parts)
        error('validateparameters:Could not find function signature')
    else
        contents = contents(parts(1):end);
    end
        
    %Extract inputs 
    signaturesearch = extractBetween( contents, "(", ")" );
    signature = signaturesearch{1};

    args = char(extractBetween(contents, "arguments", "end") );
    
    %Check for arguments blocks
    if isempty(args)
        error('Error: Pipeline Function must contain an arguments block.\n%s', ...
            'Please either add an arguments block or convert varargin to an arguments block')
    else    
        args = args(1,:); %only parse the first arguments block
    end
    
    searchstrings = strrep(strsplit( signature, "," ), " ", "");

    expressions = strcat('[\n\r]\s+',searchstrings,"\.[a-zA-Z]{1,1}[a-zA-Z0-9]+");

    matches = regexp( args, expressions, "match" );

    if ~isempty( matches )
        result = string( extractAfter( unique( horzcat(matches{:}), 'stable' ), ".") );
    else
        result = "";
    end
    
    result = result(:)';

end %validatePipelineParameters

