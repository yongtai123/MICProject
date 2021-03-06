/*globals define*/
/*jshint node:true, browser:true*/

/**
 * Generated by PluginGenerator 2.16.0 from webgme on Mon Nov 13 2017 21:29:06 GMT-0600 (CST).
 * A plugin that inherits from the PluginBase. To see source code documentation about available
 * properties and methods visit %host%/docs/source/PluginBase.html.
 */

define([
    'plugin/PluginConfig',
    'text!./metadata.json',
    'plugin/PluginBase',
    'q', //The promise library..
    'common/storage/constants' //These will be needed to check that the commit did update the branch...
], function (
    PluginConfig,
    pluginMetadata,
    PluginBase,
    Q,
    STORAGE_CONSTANTS) {
    'use strict';

    var fs = require('fs'),
        path = require('path'),
        cp = require('child_process'),
        os = require('os');
		   

    pluginMetadata = JSON.parse(pluginMetadata);

    /**
     * Initializes a new instance of SimulateMatlab.
     * @class
     * @augments {PluginBase}
     * @classdesc This class represents the plugin SimulateMatlab.
     * @constructor
     */
    var SimulateMatlab = function () {
        // Call base class' constructor.
        PluginBase.call(this);
        this.pluginMetadata = pluginMetadata;
    };

    /**
     * Metadata associated with the plugin. Contains id, name, version, description, icon, configStructue etc.
     * This is also available at the instance at this.pluginMetadata.
     * @type {object}
     */
    SimulateMatlab.metadata = pluginMetadata;

    // Prototypical inheritance from PluginBase.
    SimulateMatlab.prototype = Object.create(PluginBase.prototype);
    SimulateMatlab.prototype.constructor = SimulateMatlab;

    /**
     * Main function for the plugin to execute. This will perform the execution.
     * Notes:
     * - Always log with the provided logger.[error,warning,info,debug].
     * - Do NOT put any user interaction logic UI, etc. inside this method.
     * - callback always has to be called even if error happened.
     *
     * @param {function(string, plugin.PluginResult)} callback - the result callback
     */
    SimulateMatlab.prototype.main = function (callback) {
        // Use self to access core, project, result, logger etc from PluginBase.
        // These are all instantiated at this point.
        var self = this,
		logger = this.logger,
		activeNode = self.activeNode;

	function simulateModel(dir,modelName){
		var command;
		command = 'matlab -nodesktop -nosplash -r '+modelName+',quit';
		//command = 'python test.py';

		return Q.ninvoke(cp, 'exec', command, {cwd:dir})
			.then(function (res) {
				logger.info(res);

				return{
					dir: dir,
					//resultFilename: modelName+'.png'
					resultFilename: 'Network.jpg'
				
				};
			});
	}

        self.invokePlugin('DACodeGenerator')
            .then(function (result) {
		    if (result.getSuccess() !== true) {
			    throw new Error('DACodeGenerator did not return with success!');
		    }
		    var mFileHash = result.getArtifacts()[0];
		    self.result.addArtifact(mFileHash);
		    return self.blobClient.getObjectAsString(mFileHash);

            })
            .then(function (mFileString) {
		    logger.info('mFileString',mFileString);

		    var modelName = self.core.getAttribute(activeNode, 'name');
		    var dir = 'simplest';

		    //fs.writeFileSync(path.join(dir,'test.py'), mFileString);
		    fs.writeFileSync(path.join(dir,'getData.m'), mFileString);
		    return simulateModel(dir, modelName);

            })
            .then(function (res) {
		    logger.info('Simulate Finished!');
		    logger.info('resDir', res.dir);
		    logger.info('resFileName', res.resultFilename);
		    return self.blobClient.putFile(res.resultFilename, fs.readFileSync(path.join(res.dir,res.resultFilename)));
            })
            .then(function (csvFileHash) {
		   self.result.addArtifact(csvFileHash);
		    self.core.setAttribute(activeNode, 'simResults', csvFileHash);

		    //show image in plugin results.
		    var imgURL = self.blobClient.getRelativeViewURL(csvFileHash);
		    self.createMessage(activeNode, '<img src="'+imgURL+'"'+'width="200" height="200"/>' );
		    self.createMessage(activeNode, '<img src="'+imgURL+'"'+'width="50%" height="50%"/>' );

		    return self.save('Attached simulation results at '+ self.core.getPath(activeNode));
            })
            .then(function (commitResult) {
		    if (commitResult.status === STORAGE_CONSTANTS.SYNCED) {
                	self.result.setSuccess(true);
                	callback(null, self.result);
		    } else {
			    self.createMessage(activeNode, 'Simulation succeeded but commit did not update brach.');
			    callback(new Error('Did not update brach.'), self.result);
		    
		    }
            })
            .catch(function (err) {
                // Result success is false at invocation.
                callback(err, self.result);
            });

    };

    return SimulateMatlab;
});
