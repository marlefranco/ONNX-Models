classdef Restore < matlab.mixin.SetGet
    %RESTORE Restore a data analysis pipeline from archive files on disk
    
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
        fds 
        data
        Pipeline
    end
    
    methods
        function obj = Restore( data, location )
            %PIPELINE Construct an instance of this class
            
            arguments 
               data table
               location (1,1) string
            end
            
            obj.data = data;
            obj.location = location;
            
            obj.fds = fileDatastore( obj.location, 'ReadFcn', @obj.i_customRead, 'UniformRead', true, ...
                'FileExtensions', ".mat");
            
            pipelines = obj.fds.readall();
            obj.Pipeline = pipelines;

        end %parquet
    end %public
    
    methods
        
        function value = i_customRead( obj, file  )
            %I_CUSTOMREAD Custom read function used by fileDatastore.

            warning off
            value = mlp.Pipeline.load( obj.data, file );
            warning on
        end %i_customRead 
    end
    
    methods ( Static )
        function value = pipelines( data, location  )
            %NEW Summary of this method goes here
            %   Detailed explanation goes here
            
            io = mlp.Restore( data, location );
            
            value = io.Pipeline;
            
        end
    end %static
end %classdef 

