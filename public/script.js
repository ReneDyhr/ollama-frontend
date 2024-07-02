function searchDocuments() {
    document.getElementsByClassName('ai-results-container')[0].style.display = 'block';
    const searchTerm = document.getElementById('ai-searchInput').value;
    const resultsDiv = document.getElementById('ai-results');
    const loaderDiv = document.getElementById('ai-loader');
    loaderDiv.style.display = 'block'; // Show loader
    resultsDiv.innerHTML = ''; // Clear previous results

    if (searchTerm.trim() === '') {
        resultsDiv.innerHTML = '<p>Please enter a search term.</p>';
        return;
    }

    const options = {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        method: "POST",
        body: JSON.stringify({ "query": searchTerm })
    };

    fetch(SearchURL + '/query', options)
        .then(response => response.json())
        .then(response => {
            loaderDiv.style.display = 'none'; // Hide loader
            const resultElement = document.createElement('p');
            resultElement.innerHTML = convertLinks(markdown(response.response)).replace(new RegExp('\r?\n', 'g'), '<br />');
            resultsDiv.appendChild(resultElement);
        })
        .catch(err => console.error(err));
}

function toggleTheme() {
    const isDarkMode = document.body.classList.toggle('dark');
    document.getElementById('ai-theme-toggle').checked = isDarkMode;
    localStorage.setItem('darkMode', isDarkMode);
}

function loadTheme() {
    const darkMode = localStorage.getItem('darkMode') === 'true';
    if (darkMode) {
        document.body.classList.add('dark');
    }
    if (document.getElementById('ai-theme-toggle')) {
        document.getElementById('ai-theme-toggle').checked = darkMode;
    }
}

const convertLinks = (input) => {
   // Regular expression to match URLs, avoiding HTML tags and special characters
   const urlPattern = /(http:\/\/[^\s\)<#]+)(#[^\s\)<]*)?/g;

   // Replace URLs with <a> tags
   return input.replace(urlPattern, function(match, url, hash) {
       // Ignore the hash part by only using the `url` variable
       return `<a href="${url}" target="_blank">${url}</a>`;
   });
}
// Load the theme when the page loads
window.onload = loadTheme;