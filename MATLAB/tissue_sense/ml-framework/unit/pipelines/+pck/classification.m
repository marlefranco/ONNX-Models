classdef classification
    %CLASSIFICATION Summary of this class goes here
    %   Detailed explanation goes here
    
    methods (Static)
        
        function [value] = basePipelineClassificationNoPartition( tbl, options )
            % Data Prep function with arg blocks
            
            arguments
                tbl table
                options.Normalization(1,1) string = "zscore"
            end
            
        end %function
        
        function [value] = basePipelineClassificationNoCustomProps( tbl, options )
            % Data Prep function with arg blocks
            
            arguments
                tbl table
                options.Normalization(1,1) string = "zscore"
            end
            
        end %function
        
        function [value] = basePipelineClassification( tbl, options, other ) %#ok<*STOUT>
            % Data Prep function with arg blocks
            
            arguments
                tbl table %#ok<*INUSA>
                options.Normalization(1,1) string = "zscore"
                other.Selection
            end
            
        end %function
        
        function [value] = basePipelineNoOptions( tbl ) %#ok<*STOUT>
            
            arguments
                tbl table %#ok<*INUSA>
            end
            
        end
        
    end %methods
    
end %classdef

