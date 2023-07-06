const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8080;

http.createServer(function (request, response) {
  let filePath = '.' + request.url;
  if (filePath === './') {
    filePath = './index.html';
  }

  const extname = path.extname(filePath);
  let contentType = 'text/html';

  switch (extname) {
    case '.js':
      contentType = 'application/javascript';
      break;
    case '.css':
      contentType = 'text/css';
      break;
    case '.json':
      contentType = 'application/json';
      break;
    case '.png':
      contentType = 'image/png';
      break;
    case '.jpg':
      contentType = 'image/jpg';
      break;
    case '.wav':
      contentType = 'audio/wav';
      break;
  }

  fs.readFile(filePath, function (error, content) {
    if (error) {
      if (error.code === 'ENOENT') {
        response.writeHead(404);
        response.end('404 Not Found');
      } else {
        response.writeHead(500);
        response.end('Internal Server Error');
      }
    } else {
      response.writeHead(200, { 'Content-Type': contentType });
      response.end(content, 'utf-8');
    }
  });
}).listen(PORT);

console.log(`Server running at http://localhost:${PORT}`);
