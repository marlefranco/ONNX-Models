function R2 = coefficientOfDetermination( y, yfit, weights )

% Copyright 2021 The MathWorks Inc.

    arguments
        y       (:,1) {mustBeNumeric(y)}
        yfit    (:,1) {mustBeNumeric(yfit)}
        weights (:,1) {mustBeNumeric(weights)} = ones( size(y,1), 1)
    end
    
    sumOfWeights = sum ( weights );
    ybar         = sum( weights .* y ) / sumOfWeights;

    SST     = sum( weights .* (y - ybar).^2 );      %total sum of squares
    SSR     = sum( weights .* (yfit - ybar).^2 );   %regression sum of squares
    SSE     = sum( weights .* (y - yfit).^2 );      %residual sum of squares
    R2      = 1 - SSE / SST;

end %coefficientOfDetermination