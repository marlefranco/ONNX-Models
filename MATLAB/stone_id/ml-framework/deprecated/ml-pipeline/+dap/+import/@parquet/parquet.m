classdef parquet < matlab.mixin.SetGet
    %PARQUET Create a data analysis pipeline from parquet files on disk
    
    %   Copyright 2019 The MathWorks, Inc.
    %
    % Auth/Revision:
    %   MathWorks Consulting
    %   $Author: nhowes $
    %   $Revision: 324 $  $Date: 2019-10-29 13:16:05 -0400 (Tue, 29 Oct 2019) $
    % ---------------------------------------------------------------------
    
    
    properties 
       location
    end
    
    properties (SetAccess = private, GetAccess = public )
        pds 
        pipeline
    end
    
    methods
        function obj = parquet( location, pipelines)
            %PARQUET Construct an instance of this class
            %   Detailed explanation goes here
            
            arguments 
                location (1,1) string
                pipelines (1,1) string = ""
            end
            
            
            obj.location = location;
            
            obj.pds = parquetDatastore(...
                "s3://howessandbox/turbofanEngineDegrade/CMAPSSTrainFD001.parquet" );

            data = obj.pds.readall();
            
            if  pipelines == ""
                obj.pipeline = dap.Pipeline.new( data );
            else  
                obj.pipeline = dap.Pipeline.load( data, pipelines);
            end
            
        end %parquet
    end %public
    
    methods ( Static )
        function value = new( location, varargin  )
            %NEW Summary of this method goes here
            %   Detailed explanation goes here
            
            dIP = dap.import.parquet( location, varargin{:} );
            
            value = dIP.pipeline;
            
        end
    end %static
end %classdef 

