function [value] = basePipelineRegression( tbl, options, other ) %#ok<*STOUT,*FNDEF>
    % Data Prep function with arg blocks
    
    arguments
       tbl table %#ok<*INUSA>
       options.Normalization(1,1) string = "zscore"
       options.Engineering
       other.Selection
    end

end

%Local function 
function someLocalFunction( this, that, options )

    arguments
        this table %#ok<*INUSA>
        that
        options.Something  
    end

end