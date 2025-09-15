function mustBeValidHyperparameter( arg )

    errmsg = ['Invalid name-value argument ''OptimizeHyperparameters''.'...
        'Value must be "none" "auto" "all" or optimizeVariable type.'];

    if isstring( arg ) || iscellstr( arg ) || ischar( arg )

        if ~ismember( arg, ["none", "auto", "all"] )
            error( errmsg )
        end
    else

        if ~isa( arg,'optimizableVariable' )
            error( errmsg )
        end

    end
    
end %function
 
% Copyright 2021 The MathWorks Inc.








