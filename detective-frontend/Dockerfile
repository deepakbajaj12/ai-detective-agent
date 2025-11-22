FROM node:20-alpine
WORKDIR /app
COPY package*.json /app/
RUN npm ci || npm install
COPY . /app
EXPOSE 3000
CMD ["npm", "start", "--", "--host", "0.0.0.0"]
