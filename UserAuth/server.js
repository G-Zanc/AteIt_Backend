const express = require("express");
const morgan = require("morgan");
const helmet = require("helmet");
const cors = require("cors");
const mongoose = require("mongoose");
const multer = require('multer')
const forms = multer()

require("dotenv").config({ path: "./config.env" });
const db = process.env.ATLAS_URI;
const app = express();
const api = require('./routes')


app.use(cors({
  origin: '*'
}));
require('./services/passport')

app.use(function(req, res, next) {
  res.header("Access-Control-Allow-Origin", "*");
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, PUT, DELETE');
  res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
  if ('OPTIONS' == req.method) {
     res.sendStatus(200);
   }
   else {
     next();
}});


app.use(morgan('dev'));
app.use(express.urlencoded({extended: true}));
app.use(express.json());
app.use(helmet());

mongoose.connect(db, {
}).then(() => {
  console.log('Succefuly connected to db')
}).catch((err) => {
  console.log('err :>> ', err);
})

app.get('/', (req, res) => {
  res.send("API is live!")
})

app.use('/api/v1', api);
module.exports = app;